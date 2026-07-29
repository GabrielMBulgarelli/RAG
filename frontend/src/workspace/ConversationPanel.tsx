import { useState, type FormEvent, type KeyboardEvent } from "react";

import type { WorkspaceController } from "./useWorkspace";

interface ConversationPanelProps {
  workspace: WorkspaceController;
}

export function ConversationPanel({ workspace }: ConversationPanelProps) {
  const [draft, setDraft] = useState("");
  const documents = workspace.documentList?.documents ?? [];
  const busy = workspace.activeOperation !== null;
  const canQuery = Boolean(
    workspace.runtime?.capabilities.can_query && !busy && documents.length > 0,
  );

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    if (!canQuery || !draft.trim()) {
      return;
    }
    const question = draft;
    setDraft("");
    void workspace.submitQuestion(question);
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      submit();
    }
  };

  return (
    <section className="conversation" aria-labelledby="conversation-heading">
      <header className="conversation__header">
        <div>
          <p className="eyebrow">Grounded conversation</p>
          <h2 id="conversation-heading">Ask the corpus</h2>
        </div>
        <div className="conversation-actions">
          <button
            className="text-button"
            type="button"
            disabled={workspace.exchanges.length === 0 || busy}
            onClick={() => void workspace.clearConversation()}
          >
            Clear conversation
          </button>
          <button
            className="text-button"
            type="button"
            disabled={workspace.exchanges.length === 0 || busy}
            onClick={() => void workspace.exportConversation()}
          >
            Export conversation
          </button>
        </div>
      </header>

      <div className="conversation__scroll" aria-live="polite">
        {workspace.loadingWorkspace ? (
          <div className="workspace-empty" role="status">
            <span className="status-orbit" aria-hidden="true" />
            <h3>Checking the local workspace</h3>
            <p>Reading runtime and index state.</p>
          </div>
        ) : workspace.workspaceError ? (
          <div className="workspace-empty workspace-empty--error" role="alert">
            <h3>Workspace unavailable</h3>
            <p>{workspace.workspaceError}</p>
          </div>
        ) : workspace.exchanges.length === 0 ? (
          <div className="workspace-empty">
            <span className="empty-mark" aria-hidden="true">∷</span>
            <h3>{documents.length === 0 ? "Index local documents to begin" : "Ready for a grounded question"}</h3>
            <p>
              {documents.length === 0
                ? "Add PDF or text files in the workspace controls. Answers require indexed local evidence."
                : "Ask about the indexed corpus. Citations and retrieval details will appear beside each answer."}
            </p>
          </div>
        ) : (
          <div className="exchange-list">
            {workspace.exchanges.map((exchange) => (
              <div className="exchange" key={exchange.id}>
                <article className="message message--user" aria-label="User question">
                  <p>{exchange.question}</p>
                </article>
                {exchange.pending ? (
                  <div className="response-pending" role="status">
                    <span className="status-orbit" aria-hidden="true" />
                    Retrieving local evidence…
                  </div>
                ) : null}
                {exchange.error ? (
                  <div className="query-error" role="alert">
                    <div>
                      <strong>Answer unavailable</strong>
                      <p>{exchange.error}</p>
                    </div>
                    <button
                      className="button button--quiet"
                      type="button"
                      disabled={busy}
                      onClick={() => void workspace.retryQuestion(exchange.id)}
                    >
                      Retry question
                    </button>
                  </div>
                ) : null}
                {exchange.response ? (
                  <article
                    className="message message--assistant"
                    aria-label="Assistant response"
                    onClick={() => workspace.setSelectedExchangeId(exchange.id)}
                  >
                    <div className="answer-heading">
                      <span className={`answer-state answer-state--${exchange.response.answer_state}`}>
                        {exchange.response.answer_state}
                      </span>
                      <span>{exchange.response.diagnostics.retrieval_strategy} retrieval</span>
                    </div>
                    <p>{exchange.response.message.content}</p>
                    {exchange.response.sources.length > 0 ? (
                      <div className="citation-row" aria-label="Answer citations">
                        {exchange.response.sources.map((source) => (
                          <button
                            key={source.label}
                            className="citation"
                            type="button"
                            aria-label={`Source ${source.label}`}
                            onClick={(event) => {
                              event.stopPropagation();
                              workspace.setSelectedExchangeId(exchange.id);
                              workspace.setSelectedSourceLabel(source.label);
                            }}
                          >
                            {source.label}
                          </button>
                        ))}
                      </div>
                    ) : null}
                  </article>
                ) : null}
              </div>
            ))}
          </div>
        )}
      </div>

      <form className="composer" onSubmit={submit}>
        <label htmlFor="question">Ask about your documents</label>
        <div className="composer__input">
          <textarea
            id="question"
            rows={3}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder={documents.length === 0 ? "Index a document before asking…" : "Write a question grounded in your files…"}
            aria-keyshortcuts="Control+Enter Meta+Enter"
          />
          <button
            className="send-button"
            type="submit"
            disabled={!canQuery || !draft.trim()}
            aria-label="Send question"
          >
            Send
          </button>
        </div>
        <p className="composer__hint">
          {busy
            ? `Workspace busy: ${workspace.activeOperation?.kind.replaceAll("_", " ")}`
            : "Ctrl or ⌘ + Enter to send"}
        </p>
      </form>
    </section>
  );
}
