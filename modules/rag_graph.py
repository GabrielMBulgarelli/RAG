"""Bounded LangGraph workflow for the local RAG application."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import TypedDict, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from modules.citations import (
    build_cited_context,
    build_relevant_context,
    retain_cited_claims,
    validate_answer,
)
from modules.config import config
from modules.contracts import GraphVectorStore
from modules.models import (
    AnswerValidation,
    CitationSource,
    EvidenceDecision,
    EvidenceGrade,
    EvidenceStatus,
    QueryRefinement,
    RAGResult,
    RetrievalHit,
    RetrievalStrategy,
    Route,
    RouteDecision,
    SubqueryEvidence,
    SubquerySpec,
    TraceEvent,
)
from modules.retrieval import Retriever, select_candidates


class RAGState(TypedDict):
    query: str
    history: list[dict[str, str]]
    rewritten_query: str
    route: Route
    strategy: RetrievalStrategy
    queries: list[SubquerySpec]
    required_queries: list[SubquerySpec]
    rewritten_subqueries: list[SubquerySpec]
    preserved_hits: list[RetrievalHit]
    hits: list[RetrievalHit]
    grade: EvidenceGrade
    retry_count: int
    answer: str
    sources: list[CitationSource]
    filters: dict[str, str]
    trace: list[TraceEvent]
    result: RAGResult
    validation: AnswerValidation
    termination_hint: str


def decide_after_grading(
    status: EvidenceStatus, retry_count: int, *, has_evidence: bool = False
) -> str:
    if status in {EvidenceStatus.SUFFICIENT, EvidenceStatus.LIMITED}:
        return "answer"
    if has_evidence:
        return "abstain"
    return "retry" if retry_count < config.max_retries else "abstain"


def _text(response: object) -> str:
    return str(getattr(response, "content", response)).strip()


def is_contextual_follow_up(query: str) -> bool:
    normalized = query.strip().lower()
    contextual_patterns = (
        r"^(and|also|but|then)\b",
        r"^what about\b",
        r"\b(it|its|they|them|their|that|those|this|these|former|latter)\b",
    )
    return any(re.search(pattern, normalized) for pattern in contextual_patterns)


def deterministic_route(query: str) -> RouteDecision | None:
    """Resolve only high-confidence routes without spending an LLM call."""
    normalized = " ".join(query.strip().lower().split())
    catalog_terms = r"(?:documents?|files?|sources?|corpus)"
    if re.search(
        rf"\b(?:list|show|which|what)\b.*\b{catalog_terms}\b.*\b(?:indexed|uploaded|available|have)\b",
        normalized,
    ) or re.search(rf"^(?:list|show)\s+(?:the\s+)?{catalog_terms}\b", normalized):
        return RouteDecision(route=Route.CATALOG, strategy=RetrievalStrategy.NONE)

    complex_patterns = (
        r"\bcompare\b",
        r"\bcontrast\b",
        r"\bversus\b|\bvs\.?\b",
        r"\bdifferences?\s+between\b",
        r"\bacross\b.*\b(?:documents?|files?|articles?|reports?)\b",
        r"\b(?:documents?|files?|articles?|reports?)\b.*\band\b.*\b(?:documents?|files?|articles?|reports?)\b",
    )
    if any(re.search(pattern, normalized) for pattern in complex_patterns):
        return RouteDecision(route=Route.COMPLEX_SEARCH, strategy=RetrievalStrategy.HYBRID)

    direct_question = re.match(
        r"^(?:what\s+(?:is|are|was|were|does|did)|who|when|where|which|how\s+(?:many|much|long))\b",
        normalized,
    )
    if direct_question and not re.search(
        r"\b(?:compare|contrast|versus|vs\.?|and|or)\b", normalized
    ):
        return RouteDecision(route=Route.SIMPLE_SEARCH, strategy=RetrievalStrategy.SEMANTIC)
    return None


def _trace(state: RAGState, event: TraceEvent) -> list[TraceEvent]:
    return [*state.get("trace", []), event]


def _subquery_specs(values: list[SubquerySpec] | list[str]) -> list[SubquerySpec]:
    return [
        value if isinstance(value, SubquerySpec) else SubquerySpec(id=f"SQ{index}", text=value)
        for index, value in enumerate(values, 1)
    ]


def make_chat_model(model: str | None = None) -> ChatOllama:
    """Build an Ollama client that cannot monopolize a benchmark run."""
    return ChatOllama(
        model=model or config.llm_model,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
        num_predict=512,
        client_kwargs={"timeout": 60.0},
    )


class RAGGraph:
    def __init__(self, vector_db: GraphVectorStore, llm: ChatOllama | None = None):
        self.vector_db = vector_db
        self.llm = llm or make_chat_model()
        self.retriever = Retriever(vector_db.setup())
        self.memory: dict[str, list[dict[str, str]]] = {}
        self.graph = self._compile()

    def _structured(self, schema: type, prompt: str):
        return self.llm.with_structured_output(schema, method="json_schema").invoke(
            [SystemMessage(content=prompt)], think=False
        )

    def _rewrite(self, state: RAGState) -> dict:
        started = perf_counter()
        history = state.get("history", [])
        query = state["query"]
        should_rewrite = bool(history) and is_contextual_follow_up(query)
        if not should_rewrite:
            return {
                "rewritten_query": query,
                "retry_count": 0,
                "queries": [],
                "required_queries": [],
                "preserved_hits": [],
                "hits": [],
                "trace": _trace(
                    state,
                    TraceEvent(
                        stage="rewrite",
                        decision="not_needed",
                        llm_calls=0,
                        duration_ms=(perf_counter() - started) * 1000,
                    ),
                ),
            }
        transcript = "\n".join(f"{item['role']}: {item['content']}" for item in history[-6:])
        prompt = (
            "Rewrite the latest user question so it stands alone. Preserve its intent "
            "and named entities. Return only the rewritten question.\n\n"
            f"Conversation:\n{transcript}\nuser: {query}"
        )
        rewritten = _text(self.llm.invoke([HumanMessage(content=prompt)], think=False))
        return {
            "rewritten_query": rewritten,
            "retry_count": 0,
            "queries": [],
            "required_queries": [],
            "preserved_hits": [],
            "hits": [],
            "trace": _trace(
                state,
                TraceEvent(
                    stage="rewrite",
                    decision="rewritten",
                    llm_calls=1,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _route(self, state: RAGState) -> dict:
        started = perf_counter()
        decision = deterministic_route(state["rewritten_query"])
        llm_calls = 0
        if decision is None:
            document_count = len(self.vector_db.document_names())
            prompt = f"""Classify this request for a local document assistant.
Routes: catalog asks which documents are indexed; clarification is too ambiguous;
out_of_scope is unrelated to document QA; simple_search is one direct fact;
complex_search requires comparison, synthesis, multiple facts, or evidence from
multiple named articles or sources. Mentions of articles or publications do not
make a request catalog; catalog is only for listing the indexed corpus.
Use semantic for direct searches, hybrid for keyword-sensitive or complex searches,
and none for non-search routes. The corpus contains {document_count} documents.
Request: {state["rewritten_query"]}"""
            decision = cast(RouteDecision, self._structured(RouteDecision, prompt))
            llm_calls = 1
        strategy = decision.strategy
        if decision.route in {Route.CATALOG, Route.CLARIFICATION, Route.OUT_OF_SCOPE}:
            strategy = RetrievalStrategy.NONE
        elif decision.route == Route.COMPLEX_SEARCH:
            strategy = RetrievalStrategy.HYBRID
        return {
            "route": decision.route,
            "strategy": strategy,
            "trace": _trace(
                state,
                TraceEvent(
                    stage="route",
                    decision=f"{decision.route.value}:{strategy.value}",
                    llm_calls=llm_calls,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _after_route(self, state: RAGState) -> str:
        return state["route"].value

    def _direct(self, state: RAGState) -> dict:
        started = perf_counter()
        answer = (
            "Please clarify what document or topic you want me to search."
            if state["route"] == Route.CLARIFICATION
            else "I can only answer questions supported by the indexed documents."
        )
        return {
            "answer": answer,
            "grade": EvidenceGrade(
                status=EvidenceStatus.INSUFFICIENT,
                answer_supported=False,
                reason="No retrieval was appropriate for this route.",
            ),
            "trace": _trace(
                state,
                TraceEvent(
                    stage="direct",
                    decision=state["route"].value,
                    llm_calls=0,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _catalog(self, state: RAGState) -> dict:
        started = perf_counter()
        names = self.vector_db.document_names()
        answer = (
            "Indexed documents:\n" + "\n".join(f"- {name}" for name in names)
            if names
            else "No documents are indexed yet."
        )
        status = EvidenceStatus.SUFFICIENT if names else EvidenceStatus.INSUFFICIENT
        return {
            "answer": answer,
            "grade": EvidenceGrade(
                status=status,
                answer_supported=bool(names),
                reason="Catalog inspected.",
            ),
            "trace": _trace(
                state,
                TraceEvent(
                    stage="catalog",
                    decision=status.value,
                    llm_calls=0,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _decompose(self, state: RAGState) -> dict:
        started = perf_counter()
        queries = [SubquerySpec(id="SQ1", text=state["rewritten_query"])]
        return {
            "queries": queries,
            "required_queries": queries,
            "trace": _trace(
                state,
                TraceEvent(
                    stage="decompose",
                    decision="preserve_full_question",
                    selected_count=len(queries),
                    llm_calls=0,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _retrieve(self, state: RAGState) -> dict:
        started = perf_counter()
        queries = _subquery_specs(state.get("queries") or [state["rewritten_query"]])
        preserved_hits = state.get("preserved_hits", [])
        by_id: dict[str, RetrievalHit] = {hit.chunk_id: hit for hit in preserved_hits}
        coverage_chunk_ids: set[str] = set()
        retrieved_count = 0

        def search(query: SubquerySpec):
            return self.retriever.search(
                query.text,
                strategy=state["strategy"].value,
                semantic_k=config.semantic_candidates,
                sparse_k=config.sparse_candidates,
                fusion_limit=config.max_candidates,
                selection_limit=config.max_candidates,
                filters=state.get("filters"),
            )

        if len(queries) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(queries))) as executor:
                batches = list(executor.map(search, queries))
        else:
            batches = [search(queries[0])]

        for query, batch in zip(queries, batches, strict=True):
            retrieved_count += batch.retrieved_count
            if batch.hits:
                coverage_chunk_ids.add(batch.hits[0].chunk_id)
            for hit in batch.hits:
                hit = hit.model_copy(update={"subqueries": sorted({*hit.subqueries, query.text})})
                old = by_id.get(hit.chunk_id)
                if old is None:
                    by_id[hit.chunk_id] = hit
                else:
                    component_values = {
                        "semantic_score": max(
                            value
                            for value in (old.semantic_score, hit.semantic_score)
                            if value is not None
                        )
                        if old.semantic_score is not None or hit.semantic_score is not None
                        else None,
                        "sparse_score": max(
                            value
                            for value in (old.sparse_score, hit.sparse_score)
                            if value is not None
                        )
                        if old.sparse_score is not None or hit.sparse_score is not None
                        else None,
                        "fused_score": max(
                            value
                            for value in (old.fused_score, hit.fused_score)
                            if value is not None
                        )
                        if old.fused_score is not None or hit.fused_score is not None
                        else None,
                    }
                    by_id[hit.chunk_id] = old.model_copy(
                        update={
                            "score": max(old.score, hit.score),
                            **component_values,
                            "subqueries": sorted({*old.subqueries, *hit.subqueries}),
                        }
                    )
        selected = select_candidates(
            list(by_id.values()),
            limit=config.max_context_chunks,
            preserve_chunk_ids={
                *(hit.chunk_id for hit in preserved_hits),
                *coverage_chunk_ids,
            },
        )
        return {
            "queries": queries,
            "required_queries": state.get("required_queries") or queries,
            "hits": selected,
            "trace": _trace(
                state,
                TraceEvent(
                    stage="retrieve",
                    decision=state["strategy"].value,
                    candidate_count=retrieved_count,
                    retrieved_count=retrieved_count,
                    fused_count=len(by_id),
                    selected_count=len(selected),
                    retry_count=state.get("retry_count", 0),
                    llm_calls=0,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _grade(self, state: RAGState) -> dict:
        started = perf_counter()
        hits = state.get("hits", [])
        queries = _subquery_specs(
            state.get("required_queries") or state.get("queries") or [state["rewritten_query"]]
        )
        if not hits:
            return {
                "grade": EvidenceGrade(
                    status=EvidenceStatus.INSUFFICIENT,
                    answer_supported=False,
                    unsupported_subqueries=[query.id for query in queries],
                    reason="No evidence.",
                ),
                "trace": _trace(
                    state,
                    TraceEvent(
                        stage="grade",
                        decision=EvidenceStatus.INSUFFICIENT.value,
                        candidate_count=0,
                        retry_count=state.get("retry_count", 0),
                        llm_calls=0,
                        duration_ms=(perf_counter() - started) * 1000,
                    ),
                ),
            }
        context, _ = build_cited_context(hits)
        subquery_contract = [{"id": query.id, "text": query.text} for query in queries]
        proposed = cast(
            EvidenceDecision,
            self._structured(
                EvidenceDecision,
                f"""Grade the evidence separately for every required subquery. For each supplied
subquery ID, return one assessment containing that exact ID and its useful [C#] labels.
Do not reproduce or edit subquery text. When the evidence directly resolves the final
question, set answer_supported true and draft the final answer in drafted_answer. The draft
must begin with the answer entity or yes/no, use exactly one sentence, and cite every factual
claim with the supplied [C#] labels. Otherwise set answer_supported false and drafted_answer
to an empty string. A shared entity explicitly named across independent evidence items
directly supports an entity-linking answer; unrelated topical overlap or partial clues do not.
Evidence is sufficient only when every subquery has at least one useful label. Use limited
for partial coverage or conflicting evidence, and insufficient when the final answer is not
directly supported. Identify conflicting labels when present.
Question: {state["rewritten_query"]}
Subqueries: {subquery_contract}\nEvidence:\n{context}""",
            ),
        )
        valid_labels = {f"C{index}" for index in range(1, len(hits) + 1)}
        query_ids = [query.id for query in queries]
        labels_by_query = {query_id: [] for query_id in query_ids}
        for assessment in proposed.assessments:
            if assessment.subquery_id not in labels_by_query:
                continue
            labels_by_query[assessment.subquery_id] = list(
                dict.fromkeys(
                    label for label in assessment.relevant_labels if label in valid_labels
                )
            )
        supported = [query_id for query_id in query_ids if labels_by_query[query_id]]
        unsupported = [query_id for query_id in query_ids if query_id not in supported]
        relevant_labels = list(
            dict.fromkeys(label for query_id in query_ids for label in labels_by_query[query_id])
        )
        coverage = len(supported) / len(queries) if queries else 0.0
        conflict_labels = [label for label in proposed.conflicting_labels if label in valid_labels]
        conflict = proposed.conflict and bool(conflict_labels)
        if not supported or not proposed.answer_supported:
            status = EvidenceStatus.INSUFFICIENT
        elif coverage < 1.0 or conflict:
            status = EvidenceStatus.LIMITED
        else:
            status = EvidenceStatus.SUFFICIENT
        grade = EvidenceGrade(
            status=status,
            answer_supported=proposed.answer_supported,
            drafted_answer=(
                proposed.drafted_answer.strip()
                if status in {EvidenceStatus.SUFFICIENT, EvidenceStatus.LIMITED}
                else ""
            ),
            assessments=[
                SubqueryEvidence(
                    subquery_id=query_id,
                    relevant_labels=labels_by_query[query_id],
                )
                for query_id in query_ids
            ],
            relevant_labels=relevant_labels,
            supported_subqueries=supported,
            unsupported_subqueries=unsupported,
            relevant_labels_by_subquery=labels_by_query,
            coverage_fraction=coverage,
            fully_supported=coverage == 1.0 and not conflict,
            partially_supported=0.0 < coverage < 1.0,
            conflict=conflict,
            conflicting_labels=conflict_labels,
            reason=proposed.reason,
        )
        return {
            "grade": grade,
            "trace": _trace(
                state,
                TraceEvent(
                    stage="grade",
                    decision=grade.status.value,
                    candidate_count=len(hits),
                    selected_count=len(grade.relevant_labels),
                    retry_count=state.get("retry_count", 0),
                    llm_calls=1,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _after_grade(self, state: RAGState) -> str:
        return decide_after_grading(
            state["grade"].status,
            state["retry_count"],
            has_evidence=bool(state.get("hits")),
        )

    def _refine(self, state: RAGState) -> dict:
        started = perf_counter()
        grade = state["grade"]
        required = _subquery_specs(state.get("required_queries") or state.get("queries") or [])
        pending_ids = set(grade.unsupported_subqueries) or {query.id for query in required}
        pending = [query for query in required if query.id in pending_ids]
        relevant_labels = set(grade.relevant_labels)
        preserved_hits = [
            hit
            for index, hit in enumerate(state.get("hits", []), 1)
            if f"C{index}" in relevant_labels
        ]
        retry_count = state["retry_count"] + 1
        proposed = cast(
            QueryRefinement,
            self._structured(
                QueryRefinement,
                f"""Rewrite only the unsupported retrieval queries so they are materially
different and more likely to retrieve the missing evidence. Keep the supplied IDs as
the keys in rewrites. Return no other IDs and do not answer the question.
Question: {state.get("rewritten_query", state.get("query", ""))}
Unsupported subqueries: {[query.model_dump() for query in pending]}""",
            ),
        )
        rewritten = [
            SubquerySpec(id=query.id, text=text.strip())
            for query in pending
            if (text := proposed.rewrites.get(query.id, "")).strip()
            and " ".join(text.lower().split()) != " ".join(query.text.lower().split())
        ][: config.max_subqueries]
        decision = "rewrite_unsupported" if rewritten else "retry_noop"
        return {
            "queries": rewritten,
            "rewritten_subqueries": [
                *state.get("rewritten_subqueries", []),
                *rewritten,
            ],
            "preserved_hits": preserved_hits,
            "retry_count": retry_count,
            "termination_hint": None if rewritten else "retry_noop",
            "trace": _trace(
                state,
                TraceEvent(
                    stage="retry",
                    decision=decision,
                    selected_count=len(preserved_hits),
                    retry_count=retry_count,
                    llm_calls=1,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _answer(self, state: RAGState) -> dict:
        started = perf_counter()
        context, sources = build_relevant_context(
            state["hits"], set(state["grade"].relevant_labels)
        )
        drafted_answer = state["grade"].drafted_answer.strip()
        if drafted_answer:
            return {
                "answer": drafted_answer,
                "sources": sources,
                "trace": _trace(
                    state,
                    TraceEvent(
                        stage="generate",
                        decision="reuse_grounded_draft",
                        llm_calls=0,
                        duration_ms=(perf_counter() - started) * 1000,
                    ),
                ),
            }
        qualifier = (
            "Clearly say the answer is limited. "
            if state["grade"].status == EvidenceStatus.LIMITED
            else ""
        )
        prompt = f"""Begin with the answer entity or yes/no. Use exactly one sentence.
Answer only from the evidence. {qualifier}Cite every factual claim with
its exact [C#] label. Do not invent labels, filenames, confidence, or a sources list.
Question: {state["rewritten_query"]}\nEvidence:\n{context}"""
        return {
            "answer": _text(self.llm.invoke([HumanMessage(content=prompt)], think=False)),
            "sources": sources,
            "trace": _trace(
                state,
                TraceEvent(
                    stage="generate",
                    decision="answer",
                    llm_calls=1,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _abstain(self, state: RAGState) -> dict:
        started = perf_counter()
        return {
            "answer": "I could not find enough evidence in the indexed documents to answer.",
            "sources": [],
            "trace": _trace(
                state,
                TraceEvent(
                    stage="abstain",
                    decision="insufficient",
                    llm_calls=0,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _validate(self, state: RAGState) -> dict:
        started = perf_counter()
        requires_grounding = (
            state["route"] in {Route.SIMPLE_SEARCH, Route.COMPLEX_SEARCH}
            and state["grade"].status != EvidenceStatus.INSUFFICIENT
        )
        known_labels = {f"C{index}" for index in range(1, len(state.get("hits", [])) + 1)}
        initial = validate_answer(
            state["answer"],
            state.get("sources", []),
            known_labels=known_labels,
            require_citations=requires_grounding,
        )
        validation = initial
        answer = initial.sanitized_text
        cited = initial.used_sources
        validation_failed = False
        validation_decision = "valid"
        validation_llm_calls = 0
        if requires_grounding and not initial.is_valid:
            validation_llm_calls = 1
            context, relevant_sources = build_relevant_context(
                state.get("hits", []), set(state["grade"].relevant_labels)
            )
            labels = ", ".join(f"[{source.label}]" for source in relevant_sources)
            prompt = f"""Repair the answer using only the evidence below. Begin with a direct,
concise answer and cite every factual claim. Use only these labels: {labels}. Return
only the repaired answer.
Question: {state.get("rewritten_query", state.get("query", ""))}
Violations: {[item.value for item in initial.violations]}
Original answer: {state["answer"]}
Evidence:\n{context}"""
            repaired_text = _text(self.llm.invoke([HumanMessage(content=prompt)], think=False))
            repaired = validate_answer(
                repaired_text,
                relevant_sources,
                known_labels=known_labels,
                require_citations=True,
            )
            if repaired.is_valid:
                answer = repaired.sanitized_text
                cited = repaired.used_sources
                validation = repaired.model_copy(
                    update={
                        "repair_attempted": True,
                        "repair_succeeded": True,
                        "initial_violations": initial.violations,
                    }
                )
                validation_decision = "repaired"
            else:
                salvaged = retain_cited_claims(
                    state["answer"],
                    relevant_sources,
                    known_labels=known_labels,
                )
                if salvaged.is_valid and salvaged.used_sources:
                    answer = salvaged.sanitized_text
                    cited = salvaged.used_sources
                    validation = salvaged.model_copy(
                        update={
                            "repair_attempted": True,
                            "repair_succeeded": False,
                            "initial_violations": initial.violations,
                            "repair_violations": repaired.violations,
                        }
                    )
                    validation_decision = "salvaged"
                else:
                    answer = "I could not produce a fully cited answer from the available evidence."
                    cited = []
                    validation = validate_answer(
                        answer, [], known_labels=set(), require_citations=False
                    ).model_copy(
                        update={
                            "repair_attempted": True,
                            "repair_succeeded": False,
                            "initial_violations": initial.violations,
                            "repair_violations": repaired.violations,
                        }
                    )
                    validation_failed = True
                    validation_decision = "fallback"
        if state["route"] == Route.CLARIFICATION:
            termination = "clarification"
        elif state["route"] == Route.CATALOG:
            termination = "catalog"
        elif state["route"] == Route.OUT_OF_SCOPE:
            termination = "out_of_scope"
        elif validation_failed:
            termination = "validation_failed"
        elif state.get("termination_hint"):
            termination = state["termination_hint"]
        elif state["grade"].status == EvidenceStatus.INSUFFICIENT:
            termination = "unsupported"
        elif state["grade"].status == EvidenceStatus.LIMITED:
            termination = "limited"
        else:
            termination = "supported"
        trace = _trace(
            state,
            TraceEvent(
                stage="validate",
                decision=validation_decision,
                retry_count=state.get("retry_count", 0),
                llm_calls=validation_llm_calls,
                duration_ms=(perf_counter() - started) * 1000,
            ),
        )
        trace.append(
            TraceEvent(
                stage="terminate",
                decision=state["grade"].status.value,
                retry_count=state.get("retry_count", 0),
                llm_calls=0,
                duration_ms=0.0,
                termination=termination,
            )
        )
        required_queries = _subquery_specs(
            state.get("required_queries") or state.get("queries") or []
        )
        result = RAGResult(
            answer=answer,
            standalone_query=state.get("rewritten_query", state.get("query", "")),
            route=state["route"],
            strategy=state["strategy"],
            retry_count=state.get("retry_count", 0),
            evidence_status=state["grade"].status,
            sources=cited,
            subqueries=[query.text for query in required_queries],
            subquery_specs=required_queries,
            rewritten_subqueries=state.get("rewritten_subqueries", []),
            supported_subquery_ids=state["grade"].supported_subqueries,
            relevant_labels=state["grade"].relevant_labels,
            retrieval_hits=state.get("hits", []),
            trace=trace,
            validation=validation,
            conflict=state["grade"].conflict,
        )
        return {
            "answer": answer,
            "sources": cited,
            "validation": validation,
            "trace": trace,
            "result": result,
        }

    def _compile(self):
        graph = StateGraph(RAGState)
        for name, node in {
            "rewrite": self._rewrite,
            "route_request": self._route,
            "catalog": self._catalog,
            "direct": self._direct,
            "decompose": self._decompose,
            "retrieve": self._retrieve,
            "grade_evidence": self._grade,
            "refine": self._refine,
            "generate_answer": self._answer,
            "abstain": self._abstain,
            "validate": self._validate,
        }.items():
            graph.add_node(name, node)
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "route_request")
        graph.add_conditional_edges(
            "route_request",
            self._after_route,
            {
                "catalog": "catalog",
                "clarification": "direct",
                "out_of_scope": "direct",
                "simple_search": "retrieve",
                "complex_search": "decompose",
            },
        )
        graph.add_edge("decompose", "retrieve")
        graph.add_edge("retrieve", "grade_evidence")
        graph.add_conditional_edges(
            "grade_evidence",
            self._after_grade,
            {
                "answer": "generate_answer",
                "retry": "refine",
                "abstain": "abstain",
            },
        )
        graph.add_conditional_edges(
            "refine",
            lambda state: "retrieve" if state.get("queries") else "abstain",
            {"retrieve": "retrieve", "abstain": "abstain"},
        )
        for node in ("catalog", "direct", "generate_answer", "abstain"):
            graph.add_edge(node, "validate")
        graph.add_edge("validate", END)
        return graph.compile(checkpointer=MemorySaver())

    def process_query(
        self,
        query: str,
        session_id: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> dict:
        session_id = session_id or str(uuid4())
        history = self.memory.setdefault(session_id, [])
        state = self.graph.invoke(
            {"query": query, "history": history, "filters": filters or {}, "trace": []},
            config={"configurable": {"thread_id": session_id}},
        )
        result: RAGResult = state["result"]
        history.extend(
            [{"role": "user", "content": query}, {"role": "assistant", "content": result.answer}]
        )
        del history[:-12]
        return result.model_dump(mode="json")

    def clear(self, session_id: str) -> None:
        self.memory.pop(session_id, None)
