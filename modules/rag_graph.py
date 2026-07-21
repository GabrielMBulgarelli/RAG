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

from modules.citations import build_cited_context, build_relevant_context, validate_answer
from modules.config import config
from modules.models import (
    AnswerValidation,
    CitationSource,
    EvidenceGrade,
    EvidenceStatus,
    QueryDecomposition,
    RAGResult,
    RetrievalHit,
    RetrievalStrategy,
    Route,
    RouteDecision,
    TraceEvent,
)
from modules.retrieval import Retriever, select_candidates
from modules.vector_db import VectorDBManager


class RAGState(TypedDict):
    query: str
    history: list[dict[str, str]]
    rewritten_query: str
    route: Route
    strategy: RetrievalStrategy
    queries: list[str]
    required_queries: list[str]
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


def decide_after_grading(status: EvidenceStatus, retry_count: int) -> str:
    if status in {EvidenceStatus.SUFFICIENT, EvidenceStatus.LIMITED}:
        return "answer"
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
        return RouteDecision(
            route=Route.COMPLEX_SEARCH, strategy=RetrievalStrategy.HYBRID
        )

    direct_question = re.match(
        r"^(?:what\s+(?:is|are|was|were|does|did)|who|when|where|which|how\s+(?:many|much|long))\b",
        normalized,
    )
    if direct_question and not re.search(
        r"\b(?:compare|contrast|versus|vs\.?|and|or)\b", normalized
    ):
        return RouteDecision(
            route=Route.SIMPLE_SEARCH, strategy=RetrievalStrategy.SEMANTIC
        )
    return None


def _trace(state: RAGState, event: TraceEvent) -> list[TraceEvent]:
    return [*state.get("trace", []), event]


class RAGGraph:
    def __init__(self, vector_db: VectorDBManager, llm: ChatOllama | None = None):
        self.vector_db = vector_db
        self.llm = llm or ChatOllama(
            model=config.llm_model,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
        )
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
            "grade": EvidenceGrade(status=status, reason="Catalog inspected."),
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
        result = cast(
            QueryDecomposition,
            self._structured(
                QueryDecomposition,
                "Break this complex document question into one to four independent retrieval "
                "queries. Do not answer it.\nQuestion: " + state["rewritten_query"],
            ),
        )
        queries = result.queries[: config.max_subqueries]
        return {
            "queries": queries,
            "required_queries": queries,
            "trace": _trace(
                state,
                TraceEvent(
                    stage="decompose",
                    selected_count=len(queries),
                    llm_calls=1,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _retrieve(self, state: RAGState) -> dict:
        started = perf_counter()
        queries = state.get("queries") or [state["rewritten_query"]]
        preserved_hits = state.get("preserved_hits", [])
        by_id: dict[str, RetrievalHit] = {hit.chunk_id: hit for hit in preserved_hits}
        retrieved_count = 0

        def search(query: str):
            return self.retriever.search(
                query,
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
            for hit in batch.hits:
                hit = hit.model_copy(update={"subqueries": sorted({*hit.subqueries, query})})
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
            preserve_chunk_ids={hit.chunk_id for hit in preserved_hits},
        )
        return {
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
        queries = (
            state.get("required_queries")
            or state.get("queries")
            or [state["rewritten_query"]]
        )
        if not hits:
            return {
                "grade": EvidenceGrade(
                    status=EvidenceStatus.INSUFFICIENT,
                    unsupported_subqueries=queries,
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
        proposed = cast(
            EvidenceGrade,
            self._structured(
                EvidenceGrade,
                f"""Grade the evidence separately for every required subquery. Reproduce each
subquery exactly in supported_subqueries or unsupported_subqueries and map its relevant
[C#] labels in relevant_labels_by_subquery. Evidence is sufficient only when every
subquery is supported. Use limited for partial coverage or conflicting evidence, and
insufficient when none is useful. Identify conflicting labels when present.
Question: {state["rewritten_query"]}
Subqueries: {queries}\nEvidence:\n{context}""",
            ),
        )
        valid_labels = {f"C{index}" for index in range(1, len(hits) + 1)}
        labels_by_query = {
            query: [
                label
                for label in proposed.relevant_labels_by_subquery.get(query, [])
                if label in valid_labels
            ]
            for query in queries
        }
        if len(queries) == 1 and not labels_by_query[queries[0]]:
            labels_by_query[queries[0]] = [
                label for label in proposed.relevant_labels if label in valid_labels
            ]
        supported = [
            query
            for query in queries
            if query in proposed.supported_subqueries and labels_by_query[query]
        ]
        unsupported = [query for query in queries if query not in supported]
        relevant_labels = list(
            dict.fromkeys(label for query in queries for label in labels_by_query[query])
        )
        coverage = len(supported) / len(queries) if queries else 0.0
        conflict_labels = [label for label in proposed.conflicting_labels if label in valid_labels]
        conflict = proposed.conflict and bool(conflict_labels)
        if not supported:
            status = EvidenceStatus.INSUFFICIENT
        elif coverage < 1.0 or conflict:
            status = EvidenceStatus.LIMITED
        else:
            status = EvidenceStatus.SUFFICIENT
        grade = proposed.model_copy(
            update={
                "status": status,
                "relevant_labels": relevant_labels,
                "supported_subqueries": supported,
                "unsupported_subqueries": unsupported,
                "relevant_labels_by_subquery": labels_by_query,
                "coverage_fraction": coverage,
                "fully_supported": coverage == 1.0 and not conflict,
                "partially_supported": 0.0 < coverage < 1.0,
                "conflict": conflict,
                "conflicting_labels": conflict_labels,
            }
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
        return decide_after_grading(state["grade"].status, state["retry_count"])

    def _refine(self, state: RAGState) -> dict:
        started = perf_counter()
        grade = state["grade"]
        pending = grade.unsupported_subqueries or (
            state.get("required_queries") or state.get("queries") or []
        )
        relevant_labels = set(grade.relevant_labels)
        preserved_hits = [
            hit
            for index, hit in enumerate(state.get("hits", []), 1)
            if f"C{index}" in relevant_labels
        ]
        retry_count = state["retry_count"] + 1
        return {
            "queries": pending[: config.max_subqueries],
            "preserved_hits": preserved_hits,
            "retry_count": retry_count,
            "trace": _trace(
                state,
                TraceEvent(
                    stage="retry",
                    decision="target_unsupported",
                    selected_count=len(preserved_hits),
                    retry_count=retry_count,
                    llm_calls=0,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _answer(self, state: RAGState) -> dict:
        started = perf_counter()
        context, sources = build_relevant_context(
            state["hits"], set(state["grade"].relevant_labels)
        )
        qualifier = (
            "Clearly say the answer is limited. "
            if state["grade"].status == EvidenceStatus.LIMITED
            else ""
        )
        prompt = f"""Begin with a direct, concise answer, followed only by the synthesis needed.
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
        result = RAGResult(
            answer=answer,
            standalone_query=state.get("rewritten_query", state.get("query", "")),
            route=state["route"],
            strategy=state["strategy"],
            retry_count=state.get("retry_count", 0),
            evidence_status=state["grade"].status,
            sources=cited,
            subqueries=state.get("required_queries") or state.get("queries", []),
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
        graph.add_edge("refine", "retrieve")
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
