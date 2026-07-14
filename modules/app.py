"""Gradio interface exposing the existing local RAG services."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr

from modules.config import PROJECT_ROOT, config
from modules.evaluation import SYSTEMS, run_evaluation
from modules.rag_graph import RAGGraph
from modules.vector_db import VectorDBManager

DOCUMENT_HEADERS = [
    "Filename",
    "Source path",
    "Document ID",
    "Pages",
    "Chunks",
    "Content hash",
    "Status",
    "Last error",
]
TRACE_HEADERS = [
    "Stage",
    "Decision",
    "Candidates",
    "Selected",
    "Retry",
    "Termination",
    "Duration (ms)",
]
SCORE_HEADERS = [
    "Chunk ID",
    "Filename",
    "Page",
    "Semantic",
    "Sparse",
    "Fused",
    "Matched subqueries",
]
METRIC_NAMES = [
    "recall_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "route_accuracy",
    "strategy_accuracy",
    "retry_precision",
    "retry_recall",
    "citation_precision",
    "gold_evidence_citation_coverage",
    "abstention_accuracy",
    "conflict_accuracy",
    "termination_rate",
    "mean_latency_seconds",
    "p95_latency_seconds",
    "mean_llm_calls_per_query",
    "mean_retrieval_rounds_per_query",
]


class RAGApplication:
    def __init__(self, vector_db: VectorDBManager | None = None):
        self.vector_db = vector_db or VectorDBManager()
        self.rag_graph: RAGGraph | None = None
        self.last_errors: dict[str, str] = {}

    def initialize(self) -> None:
        self.vector_db.setup()
        self.rag_graph = RAGGraph(self.vector_db)

    def _graph(self) -> RAGGraph:
        if self.rag_graph is None:
            self.initialize()
        assert self.rag_graph is not None
        return self.rag_graph

    def document_rows(self) -> list[list[Any]]:
        manifest = self.vector_db.manifest()
        return [
            [
                record.filename,
                record.relative_path,
                record.document_id,
                record.page_count,
                record.chunk_count,
                record.content_hash[:12],
                "indexed",
                self.last_errors.get(record.document_id, ""),
            ]
            for record in sorted(manifest.documents.values(), key=lambda item: item.relative_path)
        ]

    @staticmethod
    def error_rows(errors: list[Any]) -> list[list[str]]:
        return [
            [error.document, error.operation, error.error_type, error.message] for error in errors
        ]

    def _reset_graph(self) -> None:
        self.rag_graph = None

    def index_selected(
        self, files: list[str] | None, progress: gr.Progress = gr.Progress(track_tqdm=False)
    ):
        progress(0, desc="Saving files")
        paths = self.vector_db.save_uploads(files or [])
        errors = []
        total_chunks = 0
        for index, path in enumerate(paths):
            progress((index, max(len(paths), 1)), desc=f"Parsing and chunking {path.name}")
            progress(None, desc=f"Embedding and upserting {path.name}")
            result = self.vector_db.index_document(path)
            if result.success:
                total_chunks += result.chunk_count
                self.last_errors.pop(result.document_id, None)
            elif result.error:
                errors.append(result.error)
                self.last_errors[result.document_id] = result.error.message
        progress(None, desc="Removing stale chunks and updating manifest")
        self._reset_graph()
        return (
            self.document_rows(),
            f"Indexed {len(paths) - len(errors)} document(s), {total_chunks} chunks.",
            self.error_rows(errors),
            self.readiness(),
        )

    def reindex_changed(self, progress: gr.Progress = gr.Progress(track_tqdm=False)):
        manifest = self.vector_db.manifest()
        changed: list[Path] = []
        for record in manifest.documents.values():
            path = self.vector_db.settings.sources_dir / record.relative_path
            if (
                path.exists()
                and hashlib.sha256(path.read_bytes()).hexdigest() != record.content_hash
            ):
                changed.append(path)
        errors = []
        for index, path in enumerate(changed):
            progress(
                (index, max(len(changed), 1)),
                desc=f"Parsing, chunking, embedding and upserting {path.name}",
            )
            result = self.vector_db.index_document(path)
            if result.error:
                errors.append(result.error)
                self.last_errors[result.document_id] = result.error.message
        progress(None, desc="Removing stale chunks and updating manifest")
        self._reset_graph()
        return (
            self.document_rows(),
            f"Reindexed {len(changed) - len(errors)} changed document(s).",
            self.error_rows(errors),
        )

    def delete_selected(self, document_id: str | None):
        if not document_id:
            return self.document_rows(), "Select a document ID to delete.", []
        deleted = self.vector_db.delete_document(document_id)
        self._reset_graph()
        status = (
            f"Deleted document {document_id}."
            if deleted
            else f"Document {document_id} was not found."
        )
        return self.document_rows(), status, []

    def rebuild_index(self, progress: gr.Progress = gr.Progress(track_tqdm=False)):
        progress(None, desc="Rebuilding collection, parsing and chunking documents")
        count = self.vector_db.rebuild()
        progress(None, desc="Embedding, upserting and updating manifest")
        self._reset_graph()
        return self.document_rows(), f"Rebuilt complete index with {count} chunks.", []

    def _ollama_info(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(
                f"{config.ollama_base_url}/api/tags", timeout=2
            ) as response:  # noqa: S310
                payload = json.load(response)
            return {
                "reachable": True,
                "models": [item.get("name", "") for item in payload.get("models", [])],
            }
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return {"reachable": False, "models": []}

    def readiness(self) -> str:
        info = self._ollama_info()
        try:
            chunks = self.vector_db.chunk_count()
            documents = len(self.vector_db.manifest().documents)
        except Exception:
            chunks, documents = 0, 0
        return f"Ollama: **{'ready' if info['reachable'] else 'unavailable'}** · Index: **{chunks} chunks** · Documents: **{documents}**"

    @staticmethod
    def source_rows(result: dict[str, Any]) -> list[list[Any]]:
        hits = {hit.get("chunk_id"): hit for hit in result.get("retrieval_hits", [])}
        return [
            [
                source.get("label"),
                source.get("filename"),
                source.get("page"),
                source.get("excerpt", ""),
                hits.get(source.get("chunk_id"), {}).get("semantic_score"),
                hits.get(source.get("chunk_id"), {}).get("sparse_score"),
                hits.get(source.get("chunk_id"), {}).get("fused_score"),
            ]
            for source in result.get("sources", [])
        ]

    @staticmethod
    def trace_rows(result: dict[str, Any]) -> list[list[Any]]:
        return [
            [
                event.get("stage", ""),
                event.get("decision") or "",
                event.get("candidate_count") if event.get("candidate_count") is not None else "",
                event.get("selected_count") if event.get("selected_count") is not None else "",
                event.get("retry_count", 0),
                event.get("termination") or "",
                event.get("duration_ms", 0.0),
            ]
            for event in result.get("trace", [])
        ]

    @staticmethod
    def score_rows(result: dict[str, Any]) -> list[list[Any]]:
        return [
            [
                hit.get("chunk_id"),
                hit.get("filename"),
                hit.get("page"),
                hit.get("semantic_score"),
                hit.get("sparse_score"),
                hit.get("fused_score"),
                ", ".join(hit.get("subqueries", [])),
            ]
            for hit in result.get("retrieval_hits", [])
        ]

    def chat(self, message: str, history: list[dict[str, str]], session_id: str):
        if not message.strip():
            return "", history, {}, "Enter a question.", [], [], []
        result = self._graph().process_query(message.strip(), session_id)
        result.setdefault("standalone_query", message.strip())
        result["original_question"] = message.strip()
        diagnostics = (
            f"Original: **{message.strip()}** · Standalone: **{result['standalone_query']}** · "
            f"Route: **{result['route']}** · Strategy: **{result['strategy']}** · "
            f"Subqueries: **{', '.join(result.get('subqueries', [])) or 'none'}** · "
            f"Retrieval rounds: **{sum(e.get('stage') == 'retrieve' for e in result.get('trace', []))}** · "
            f"Retries: **{result['retry_count']}** · Evidence: **{result['evidence_status']}** · "
            f"Conflict: **{result.get('conflict', 'not reported')}**"
        )
        messages = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["answer"]},
        ]
        return (
            "",
            messages,
            result,
            diagnostics,
            self.source_rows(result),
            self.score_rows(result),
            self.trace_rows(result),
        )

    def clear(self, session_id: str):
        if self.rag_graph is not None:
            self.rag_graph.clear(session_id)
        return [], {}, "Conversation cleared.", [], [], []

    @staticmethod
    def public_export(messages: list[Any], result: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": messages,
            "standalone_query": result.get("standalone_query"),
            "route": result.get("route"),
            "strategy": result.get("strategy"),
            "subqueries": result.get("subqueries", []),
            "retry_count": result.get("retry_count", 0),
            "evidence_status": result.get("evidence_status"),
            "citations": result.get("sources", []),
            "public_trace": result.get("trace", []),
        }

    def export_chat(self, messages: list[Any], result: dict[str, Any]) -> str:
        target = (
            config.data_dir
            / "exports"
            / f"conversation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.public_export(messages, result), indent=2) + "\n", encoding="utf-8"
        )
        return str(target)

    @staticmethod
    def load_evaluation_result(path: Path):
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        metrics = [
            [system, *[values.get(name) for name in METRIC_NAMES]]
            for system, values in summary.get("metrics", {}).items()
        ]
        failures = []
        for line in (path / "cases.jsonl").read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            if case.get("failure_labels"):
                failures.append(
                    [
                        case.get("case_id"),
                        case.get("system"),
                        case.get("route"),
                        case.get("strategy"),
                        ", ".join(case["failure_labels"]),
                    ]
                )
        run_id = summary.get("configuration", {}).get("run_id", path.name)
        return metrics, failures, f"Loaded evaluation **{run_id}** from `{path}`."

    @staticmethod
    def latest_evaluation() -> Path | None:
        root = PROJECT_ROOT / "evals" / "results"
        return max(
            (path for path in root.glob("*") if (path / "summary.json").exists()),
            default=None,
            key=lambda path: path.stat().st_mtime,
        )

    def load_latest_evaluation(self):
        latest = self.latest_evaluation()
        return (
            self.load_evaluation_result(latest)
            if latest
            else ([], [], "No stored evaluation result exists yet.")
        )

    def run_evaluation_ui(self, split: str, systems: list[str] | str):
        dataset = PROJECT_ROOT / "evals" / "mvp_cases.jsonl"
        if "REVIEW_REQUIRED_" in dataset.read_text(encoding="utf-8"):
            return (
                [],
                [],
                "Replace `REVIEW_REQUIRED_*` gold chunk IDs after indexing the reviewed corpus before live evaluation.",
            )
        selected = list(SYSTEMS) if systems == "all" or "all" in systems else list(systems)
        try:
            output = run_evaluation(dataset, selected, split)  # type: ignore[arg-type]
        except (RuntimeError, ValueError) as exc:
            return [], [], f"Evaluation could not run: {exc}"
        return self.load_evaluation_result(output)

    def diagnostic_rows(self, ollama: dict[str, Any] | None = None) -> list[list[str]]:
        info = ollama or self._ollama_info()
        model_names = {name.split(":")[0] for name in info["models"]}
        rows = [
            [
                "Ollama connectivity",
                "ok" if info["reachable"] else "error",
                "reachable" if info["reachable"] else f"Start Ollama at {config.ollama_base_url}",
            ]
        ]
        for label, model in (
            ("Chat model", config.llm_model),
            ("Embedding model", config.embedding_model),
        ):
            available = model.split(":")[0] in model_names
            rows.append(
                [
                    label,
                    "ok" if available else "error",
                    model if available else f"Missing {model}; run: ollama pull {model}",
                ]
            )
        try:
            manifest = self.vector_db.manifest()
            reconciliation = self.vector_db.reconcile_index()
            rows.extend(
                [
                    ["Chroma collection", "ok", f"{self.vector_db.chunk_count()} chunks"],
                    ["Manifest", "ok", f"valid; {len(manifest.documents)} documents"],
                    [
                        "Missing Chroma chunks",
                        "ok" if not reconciliation.missing_chunk_ids else "error",
                        str(len(reconciliation.missing_chunk_ids)),
                    ],
                    [
                        "Orphan Chroma chunks",
                        "ok" if not reconciliation.orphan_chunk_ids else "warning",
                        str(len(reconciliation.orphan_chunk_ids)),
                    ],
                    [
                        "Duplicate IDs",
                        "ok" if not reconciliation.duplicate_chunk_ids else "error",
                        str(len(reconciliation.duplicate_chunk_ids)),
                    ],
                    [
                        "Missing source files",
                        "ok" if not reconciliation.missing_source_files else "warning",
                        str(len(reconciliation.missing_source_files)),
                    ],
                    [
                        "Index configuration",
                        "ok" if not reconciliation.incompatible_document_ids else "error",
                        "compatible"
                        if not reconciliation.incompatible_document_ids
                        else f"{len(reconciliation.incompatible_document_ids)} incompatible documents",
                    ],
                ]
            )
        except Exception as exc:
            rows.append(["Index diagnostics", "error", f"{type(exc).__name__}: {exc}"])
        try:
            if self.rag_graph is None:
                self.initialize()
            rows.append(["Graph compilation", "ok", "compiled"])
        except Exception as exc:
            rows.append(["Graph compilation", "error", f"{type(exc).__name__}: {exc}"])
        latest = self.latest_evaluation()
        rows.append(
            [
                "Latest evaluation",
                "ok" if latest else "pending",
                str(latest) if latest else "No stored result",
            ]
        )
        return rows

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="Local Document RAG") as interface:
            session_id, latest_result = gr.State(lambda: str(uuid4())), gr.State({})
            gr.Markdown(
                "# Local Document RAG\nManage the local corpus, ask grounded questions, and inspect public evidence."
            )
            with gr.Tab("Documents"):
                readiness = gr.Markdown("Use Refresh diagnostics to check readiness.")
                files = gr.File(
                    label="PDF/TXT uploads",
                    file_count="multiple",
                    file_types=[".pdf", ".txt"],
                    type="filepath",
                )
                with gr.Row():
                    index_button = gr.Button("Index selected uploads", variant="primary")
                    reindex_button = gr.Button("Reindex changed documents")
                    rebuild_button = gr.Button("Rebuild complete index")
                documents = gr.Dataframe(
                    headers=DOCUMENT_HEADERS, value=self.document_rows(), interactive=False
                )
                selected_id = gr.Dropdown(
                    label="Document ID to delete", choices=[row[2] for row in self.document_rows()]
                )
                with gr.Row():
                    delete_button = gr.Button("Delete selected document")
                    refresh_button = gr.Button("Refresh document status")
                ingestion_status = gr.Markdown()
                errors = gr.Dataframe(
                    headers=["Document", "Operation", "Error type", "Message"], interactive=False
                )
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(label="Chat", type="messages", allow_tags=False)
                message = gr.Textbox(label="Question", placeholder="Ask about your documents")
                with gr.Row():
                    send, clear = (
                        gr.Button("Send", variant="primary"),
                        gr.Button("Clear conversation"),
                    )
                    export = gr.Button("Export conversation JSON")
                export_file = gr.File(label="Public conversation export")
                evidence = gr.Markdown()
                gr.Markdown("### Sources and retrieval scores")
                sources = gr.Dataframe(
                    headers=[
                        "Citation",
                        "Filename",
                        "Page",
                        "Excerpt",
                        "Semantic",
                        "Sparse",
                        "Fused",
                    ],
                    interactive=False,
                )
                scores = gr.Dataframe(headers=SCORE_HEADERS, interactive=False)
                gr.Markdown("### Public trace")
                trace = gr.Dataframe(headers=TRACE_HEADERS, interactive=False)
            with gr.Tab("Evaluation"):
                split = gr.Dropdown(["development", "test"], value="development", label="Split")
                systems = gr.CheckboxGroup(
                    [*SYSTEMS, "all"], value=["dense", "bm25", "hybrid", "agentic"], label="Systems"
                )
                with gr.Row():
                    run_eval, load_eval = (
                        gr.Button("Run evaluation", variant="primary"),
                        gr.Button("Load latest result"),
                    )
                eval_status = gr.Markdown()
                metrics = gr.Dataframe(headers=["System", *METRIC_NAMES], interactive=False)
                failures = gr.Dataframe(
                    headers=["Case", "System", "Route", "Strategy", "Failure labels"],
                    interactive=False,
                )
            with gr.Tab("Diagnostics"):
                refresh_diagnostics = gr.Button("Refresh diagnostics")
                diagnostics = gr.Dataframe(
                    headers=["Check", "Status", "Details"], interactive=False
                )

            index_button.click(
                self.index_selected, files, [documents, ingestion_status, errors, readiness]
            )
            reindex_button.click(self.reindex_changed, None, [documents, ingestion_status, errors])
            rebuild_button.click(self.rebuild_index, None, [documents, ingestion_status, errors])
            delete_button.click(
                self.delete_selected, selected_id, [documents, ingestion_status, errors]
            )
            refresh_button.click(
                lambda: (self.document_rows(), self.readiness()), None, [documents, readiness]
            )
            for trigger in (send.click, message.submit):
                trigger(
                    self.chat,
                    [message, chatbot, session_id],
                    [message, chatbot, latest_result, evidence, sources, scores, trace],
                )
            clear.click(
                self.clear, session_id, [chatbot, latest_result, evidence, sources, scores, trace]
            )
            export.click(self.export_chat, [chatbot, latest_result], export_file)
            run_eval.click(
                self.run_evaluation_ui, [split, systems], [metrics, failures, eval_status]
            )
            load_eval.click(self.load_latest_evaluation, None, [metrics, failures, eval_status])
            refresh_diagnostics.click(self.diagnostic_rows, None, diagnostics)
        return interface


app = RAGApplication()


def main() -> int:
    app.create_interface().launch(
        server_name=config.gradio_host, server_port=config.gradio_port, share=config.gradio_share
    )
    return 0


if __name__ == "__main__":
    main()
