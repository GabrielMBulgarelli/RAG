"""Bounded LangGraph workflow for the local RAG application."""

from __future__ import annotations

import re
from time import perf_counter
from typing import TypedDict, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from modules.citations import build_cited_context, validate_citations
from modules.config import config
from modules.models import (
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
    hits: list[RetrievalHit]
    grade: EvidenceGrade
    retry_count: int
    answer: str
    sources: list[CitationSource]
    filters: dict[str, str]
    trace: list[TraceEvent]
    result: RAGResult


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
                "trace": _trace(
                    state,
                    TraceEvent(
                        stage="rewrite",
                        decision="not_needed",
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
            "trace": _trace(
                state,
                TraceEvent(
                    stage="rewrite",
                    decision="rewritten",
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _route(self, state: RAGState) -> dict:
        started = perf_counter()
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
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _after_route(self, state: RAGState) -> str:
        return state["route"].value

    def _direct(self, state: RAGState) -> dict:
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
        }

    def _catalog(self, _state: RAGState) -> dict:
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
            "trace": _trace(
                state,
                TraceEvent(
                    stage="decompose",
                    selected_count=len(queries),
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _retrieve(self, state: RAGState) -> dict:
        started = perf_counter()
        queries = state.get("queries") or [state["rewritten_query"]]
        by_id: dict[str, RetrievalHit] = {}
        retrieved_count = 0
        for query in queries:
            batch = self.retriever.search(
                query,
                strategy=state["strategy"].value,
                semantic_k=config.semantic_candidates,
                sparse_k=config.sparse_candidates,
                fusion_limit=config.max_candidates,
                selection_limit=config.max_candidates,
                filters=state.get("filters"),
            )
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
        selected = select_candidates(list(by_id.values()), limit=config.max_context_chunks)
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
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _grade(self, state: RAGState) -> dict:
        started = perf_counter()
        hits = state.get("hits", [])
        if not hits:
            return {
                "grade": EvidenceGrade(status=EvidenceStatus.INSUFFICIENT, reason="No evidence."),
                "trace": _trace(
                    state,
                    TraceEvent(
                        stage="grade",
                        decision=EvidenceStatus.INSUFFICIENT.value,
                        candidate_count=0,
                        retry_count=state.get("retry_count", 0),
                        duration_ms=(perf_counter() - started) * 1000,
                    ),
                ),
            }
        context, _ = build_cited_context(hits)
        grade = cast(
            EvidenceGrade,
            self._structured(
                EvidenceGrade,
                f"""Grade whether the evidence answers the question. Use sufficient for full
support, limited for a useful partial answer, and insufficient otherwise. List only
relevant [C#] labels. Question: {state["rewritten_query"]}\nEvidence:\n{context}""",
            ),
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
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _after_grade(self, state: RAGState) -> str:
        return decide_after_grading(state["grade"].status, state["retry_count"])

    def _refine(self, state: RAGState) -> dict:
        started = perf_counter()
        prompt = (
            "Rewrite as one more precise retrieval query. Return only the query.\n"
            + state["rewritten_query"]
        )
        refined = _text(self.llm.invoke([HumanMessage(content=prompt)], think=False))
        retry_count = state["retry_count"] + 1
        return {
            "queries": [refined],
            "retry_count": retry_count,
            "trace": _trace(
                state,
                TraceEvent(
                    stage="retry",
                    decision="refine_query",
                    retry_count=retry_count,
                    duration_ms=(perf_counter() - started) * 1000,
                ),
            ),
        }

    def _answer(self, state: RAGState) -> dict:
        context, sources = build_cited_context(state["hits"])
        qualifier = (
            "Clearly say the answer is limited. "
            if state["grade"].status == EvidenceStatus.LIMITED
            else ""
        )
        prompt = f"""Answer only from the evidence. {qualifier}Cite every factual claim with
its exact [C#] label. Do not invent labels, filenames, confidence, or a sources list.
Question: {state["rewritten_query"]}\nEvidence:\n{context}"""
        return {
            "answer": _text(self.llm.invoke([HumanMessage(content=prompt)], think=False)),
            "sources": sources,
        }

    def _abstain(self, _state: RAGState) -> dict:
        return {
            "answer": "I could not find enough evidence in the indexed documents to answer.",
            "sources": [],
        }

    def _validate(self, state: RAGState) -> dict:
        started = perf_counter()
        answer, cited = validate_citations(state["answer"], state.get("sources", []))
        if state["route"] == Route.CLARIFICATION:
            termination = "clarification"
        elif state["route"] == Route.CATALOG:
            termination = "catalog"
        elif state["route"] == Route.OUT_OF_SCOPE:
            termination = "out_of_scope"
        elif state["grade"].status == EvidenceStatus.INSUFFICIENT:
            termination = "unsupported"
        elif state["grade"].status == EvidenceStatus.LIMITED:
            termination = "limited"
        else:
            termination = "supported"
        trace = _trace(
            state,
            TraceEvent(
                stage="terminate",
                decision=state["grade"].status.value,
                retry_count=state.get("retry_count", 0),
                duration_ms=(perf_counter() - started) * 1000,
                termination=termination,
            ),
        )
        result = RAGResult(
            answer=answer,
            standalone_query=state.get("rewritten_query", state.get("query", "")),
            route=state["route"],
            strategy=state["strategy"],
            retry_count=state.get("retry_count", 0),
            evidence_status=state["grade"].status,
            sources=cited,
            subqueries=state.get("queries", []),
            retrieval_hits=state.get("hits", []),
            trace=trace,
        )
        return {"answer": answer, "sources": cited, "trace": trace, "result": result}

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
