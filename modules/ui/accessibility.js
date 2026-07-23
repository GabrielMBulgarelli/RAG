() => {
  const regionLabels = {
    "inspector-document-inventory": "Indexed documents",
    "inspector-indexing-errors": "Indexing errors",
    "ask-chatbot": "Conversation",
    "ask-sources": "Cited evidence",
    "ask-retrieval": "Retrieval scores",
    "ask-raw-trace": "Retrieval trace",
    "system-diagnostics": "System diagnostics",
  };

  const clearScrollSemantics = (target) => {
    target.removeAttribute("role");
    target.removeAttribute("aria-label");
    target.removeAttribute("tabindex");
    delete target.dataset.overflowX;
    delete target.dataset.overflowY;
  };

  const syncRegion = (region) => {
    const target = region.querySelector(".result-scroll, .wrap") || region;
    const previousTarget = region._ragScrollTarget;
    if (previousTarget && previousTarget !== target) clearScrollSemantics(previousTarget);
    region._ragScrollTarget = target;

    const overflowX = target.scrollWidth > target.clientWidth + 1;
    const overflowY = target.scrollHeight > target.clientHeight + 1;
    const nextOverflowX = String(overflowX);
    const nextOverflowY = String(overflowY);
    if (target.dataset.overflowX !== nextOverflowX) {
      target.dataset.overflowX = nextOverflowX;
    }
    if (target.dataset.overflowY !== nextOverflowY) {
      target.dataset.overflowY = nextOverflowY;
    }

    if (overflowX || overflowY) {
      const directions = [
        overflowX && "horizontally scrollable",
        overflowY && "vertically scrollable",
      ]
        .filter(Boolean)
        .join(" and ");
      const label = `${regionLabels[region.id] || "Scrollable results"}, ${directions} scrolling available`;
      target.setAttribute("role", "region");
      target.setAttribute("aria-label", label);
      target.setAttribute("tabindex", "0");
    } else {
      target.removeAttribute("role");
      target.removeAttribute("aria-label");
      target.removeAttribute("tabindex");
    }
    return target;
  };

  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const region = entry.target.closest(".overflow-region");
      if (region) syncRegion(region);
    }
  });

  const syncAccordion = (accordion) => {
    const trigger = accordion.querySelector("button.label-wrap");
    if (!trigger) return;
    trigger.setAttribute(
      "aria-expanded",
      String(accordion.classList.contains("open")),
    );
  };

  const enhanceNode = (node) => {
    if (!(node instanceof Element)) return;
    const regions = node.matches(".overflow-region")
      ? [node]
      : Array.from(node.querySelectorAll(".overflow-region"));
    for (const region of regions) {
      const target = syncRegion(region);
      if (!target.dataset.resizeObserved) {
        target.dataset.resizeObserved = "true";
        resizeObserver.observe(target);
      }
    }
    const accordions = node.matches(".gradio-accordion")
      ? [node]
      : Array.from(node.querySelectorAll(".gradio-accordion"));
    for (const accordion of accordions) syncAccordion(accordion);
  };

  enhanceNode(document.body);
  const pending = new Set();
  let frame = null;
  new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      const parentRegion = mutation.target instanceof Element
        ? mutation.target.closest(".overflow-region")
        : null;
      if (parentRegion) pending.add(parentRegion);
      for (const node of mutation.addedNodes) pending.add(node);
    }
    if (frame !== null) return;
    frame = requestAnimationFrame(() => {
      for (const node of pending) enhanceNode(node);
      pending.clear();
      frame = null;
    });
  }).observe(document.body, {
    childList: true,
    subtree: true,
  });

  requestAnimationFrame(() => enhanceNode(document.body));
}
