(() => {
  const pdfInput = document.getElementById("pdfInput");
  const pdfSelection = document.getElementById("pdfSelection");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadResult = document.getElementById("uploadResult");

  const urlInput = document.getElementById("urlInput");
  const urlBtn = document.getElementById("urlIngestBtn");
  const urlResult = document.getElementById("urlResult");

  const statsBtn = document.getElementById("refreshStatsBtn");
  const statsBox = document.getElementById("statsBox");

  const questionInput = document.getElementById("questionInput");
  const askBtn = document.getElementById("askBtn");
  const chatLog = document.getElementById("chatLog");
  const particleCanvas = document.getElementById("particleCanvas");
  const pCtx = particleCanvas?.getContext("2d");
  const floatPanels = document.querySelectorAll(".float-panel");
  const particles = [];
  let lastParticleAt = 0;

  const csrf = window.CSRF_TOKEN;

  function setLoading(btn, loading) {
    btn.disabled = loading;
    btn.classList.toggle("is-loading", loading);
  }

  function setResult(el, text, state = "info") {
    if (!el) {
      return;
    }
    el.textContent = text;
    el.classList.remove("ok", "error");
    if (state === "ok") {
      el.classList.add("ok");
    } else if (state === "error") {
      el.classList.add("error");
    }
  }

  function extractErrorMessage(payload) {
    if (!payload) {
      return "Unknown error.";
    }
    if (typeof payload === "string") {
      return payload;
    }
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (Array.isArray(payload.detail)) {
      return payload.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
    try {
      return JSON.stringify(payload);
    } catch (_) {
      return "Unexpected error payload.";
    }
  }

  function formatScore(value) {
    const asNumber = Number(value);
    if (Number.isFinite(asNumber)) {
      return asNumber.toFixed(3);
    }
    return String(value ?? "n/a");
  }

  function formatBytes(bytes) {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) {
      return "0 B";
    }

    const units = ["B", "KB", "MB", "GB", "TB"];
    let size = value;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex += 1;
    }

    const formatted = size >= 100 || unitIndex === 0 ? Math.round(size) : size.toFixed(1);
    return `${formatted} ${units[unitIndex]}`;
  }

  function renderStats(payload = {}, state = "ready") {
    if (!statsBox) {
      return;
    }

    if (state === "loading") {
      statsBox.innerHTML = `
        <div class="stats-empty">
          <span class="stats-title">Loading session stats...</span>
          <p class="stats-copy">Refreshing the current workspace footprint.</p>
        </div>
      `;
      statsBox.classList.remove("has-error");
      return;
    }

    if (state === "error") {
      const message = typeof payload === "string" ? payload : "Unable to load session stats.";
      statsBox.innerHTML = `
        <div class="stats-empty">
          <span class="stats-title">Stats unavailable</span>
          <p class="stats-copy">${message}</p>
        </div>
      `;
      statsBox.classList.add("has-error");
      return;
    }

    const fileCount = Number(payload.pdf_files ?? payload.file_count ?? 0);
    const totalBytes = Number(payload.bytes ?? 0);
    const hasData = fileCount > 0 || totalBytes > 0;

    statsBox.classList.remove("has-error");

    if (!hasData) {
      statsBox.innerHTML = `
        <div class="stats-empty">
          <span class="stats-title">No session data yet</span>
          <p class="stats-copy">Upload a PDF or ingest a URL to start building this workspace.</p>
        </div>
      `;
      return;
    }

    statsBox.innerHTML = `
      <div class="stats-grid">
        <article class="stat-tile">
          <span class="stat-label">Files stored</span>
          <strong class="stat-value">${fileCount}</strong>
        </article>
        <article class="stat-tile">
          <span class="stat-label">Storage used</span>
          <strong class="stat-value">${formatBytes(totalBytes)}</strong>
        </article>
      </div>
      <p class="stats-footnote">Current session footprint across uploaded and indexed user files.</p>
    `;
  }

  function updatePdfSelection() {
    const files = Array.from(pdfInput?.files || []);
    if (!pdfSelection) {
      return;
    }
    if (!files.length) {
      pdfSelection.textContent = "No PDFs selected.";
      return;
    }
    if (files.length === 1) {
      pdfSelection.textContent = `1 PDF selected: ${files[0].name}`;
      return;
    }
    const preview = files
      .slice(0, 3)
      .map((f) => f.name)
      .join(", ");
    const moreCount = files.length - 3;
    pdfSelection.textContent =
      moreCount > 0
        ? `${files.length} PDFs selected: ${preview}, +${moreCount} more`
        : `${files.length} PDFs selected: ${preview}`;
  }

  function initButtonAnimations() {
    const buttons = document.querySelectorAll(".btn");
    buttons.forEach((btn) => {
      btn.addEventListener("click", (event) => {
        if (!btn || btn.disabled) {
          return;
        }
        const rect = btn.getBoundingClientRect();
        const ripple = document.createElement("span");
        const size = Math.max(rect.width, rect.height) * 1.35;
        ripple.className = "ripple";
        ripple.style.width = `${size}px`;
        ripple.style.height = `${size}px`;
        ripple.style.left = `${event.clientX - rect.left - size / 2}px`;
        ripple.style.top = `${event.clientY - rect.top - size / 2}px`;
        btn.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);
      });
    });
  }

  function initFloatingPanels() {
    floatPanels.forEach((panel) => {
      panel.addEventListener("click", () => {
        panel.classList.toggle("is-pinned");
      });
    });
  }

  function resizeParticlesCanvas() {
    if (!particleCanvas) {
      return;
    }
    particleCanvas.width = window.innerWidth;
    particleCanvas.height = window.innerHeight;
  }

  function spawnParticles(x, y, burst = false) {
    if (!pCtx) {
      return;
    }

    const now = performance.now();
    if (!burst && now - lastParticleAt < 24) {
      return;
    }
    lastParticleAt = now;

    const count = burst ? 4 : 2;
    for (let i = 0; i < count; i += 1) {
      particles.push({
        x,
        y,
        vx: (Math.random() - 0.5) * 0.8,
        vy: (Math.random() - 0.5) * 0.8,
        life: burst ? 34 + Math.random() * 12 : 24 + Math.random() * 8,
        maxLife: burst ? 44 : 30,
        size: 0.8 + Math.random() * 1.2,
        r: 186 + Math.floor(Math.random() * 26),
        g: 164 + Math.floor(Math.random() * 22),
        b: 136 + Math.floor(Math.random() * 18),
      });
    }
  }

  function renderParticles() {
    if (!pCtx || !particleCanvas) {
      return;
    }

    pCtx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);

    for (let i = particles.length - 1; i >= 0; i -= 1) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.002;
      p.life -= 1;

      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }

      pCtx.fillStyle = `rgba(${p.r}, ${p.g}, ${p.b}, ${p.life / p.maxLife})`;
      pCtx.beginPath();
      pCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      pCtx.fill();
    }
  }

  function animateParticles() {
    renderParticles();
    requestAnimationFrame(animateParticles);
  }

  function initCursorEffects() {
    resizeParticlesCanvas();
    animateParticles();

    window.addEventListener("mousemove", (event) => {
      document.documentElement.style.setProperty("--mx", `${event.clientX}px`);
      document.documentElement.style.setProperty("--my", `${event.clientY}px`);
      spawnParticles(event.clientX, event.clientY);
    });

    const hoverTargets = document.querySelectorAll(
      "button, a, input, textarea, .feature-card, .panel, .float-panel, details summary"
    );

    hoverTargets.forEach((el) => {
      el.addEventListener("mouseenter", () => {
        document.body.classList.add("cursor-active");
        const rect = el.getBoundingClientRect();
        spawnParticles(rect.left + rect.width / 2, rect.top + rect.height / 2, true);
      });
      el.addEventListener("mouseleave", () => document.body.classList.remove("cursor-active"));
    });

    window.addEventListener("mousedown", (event) => {
      document.body.classList.add("cursor-click");
      spawnParticles(event.clientX, event.clientY, true);
      setTimeout(() => document.body.classList.remove("cursor-click"), 120);
    });

    window.addEventListener("resize", resizeParticlesCanvas);
  }

  function appendMessage(type, text, retrieval = []) {
    const div = document.createElement("div");
    div.className = `msg ${type}`;
    div.textContent = text;

    if (type === "bot" && Array.isArray(retrieval) && retrieval.length > 0) {
      const details = document.createElement("details");
      details.className = "retrieval-details";

      const summary = document.createElement("summary");
      summary.textContent = `Show retrieval details (${retrieval.length})`;
      details.appendChild(summary);

      const list = document.createElement("ul");
      list.className = "retrieval-list";

      retrieval.slice(0, 5).forEach((item, idx) => {
        const li = document.createElement("li");
        const label = item.label || `Chunk ${idx + 1}`;
        const score = formatScore(item.score);
        const source = item.source || "unknown source";
        li.textContent = `${idx + 1}. ${label} | score ${score} | ${source}`;
        list.appendChild(li);
      });

      details.appendChild(list);
      div.appendChild(details);
    }

    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  async function handleUpload() {
    const files = Array.from(pdfInput.files || []);
    if (!files.length) {
      setResult(uploadResult, "Select at least one PDF first.", "error");
      return;
    }

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));

    setLoading(uploadBtn, true);
    setResult(uploadResult, "Uploading + ingesting PDFs...");

    try {
      const resp = await fetch("/api/upload-pdfs/", {
        method: "POST",
        body: formData,
        headers: { "X-CSRFToken": csrf },
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(extractErrorMessage(data.error || data));
      }

      const ingest = data.data.ingest;
      setResult(
        uploadResult,
        `Ingested ${ingest.files_ingested}/${ingest.files_total} PDFs, added ${ingest.chunks_added} chunks.`,
        "ok"
      );
      pdfInput.value = "";
      updatePdfSelection();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setResult(uploadResult, `Upload failed: ${message}`, "error");
    } finally {
      setLoading(uploadBtn, false);
    }
  }

  async function handleUrlIngest() {
    const url = (urlInput.value || "").trim();
    if (!url) {
      setResult(urlResult, "Enter a URL first.", "error");
      return;
    }

    setLoading(urlBtn, true);
    setResult(urlResult, "Ingesting URL...");

    try {
      const resp = await fetch("/api/ingest-url/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": csrf,
        },
        body: JSON.stringify({ url }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(extractErrorMessage(data.error || data));
      }

      setResult(urlResult, `Added ${data.data.chunks_added} chunks from ${data.data.url}.`, "ok");
      urlInput.value = "";
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setResult(urlResult, `URL ingest failed: ${message}`, "error");
    } finally {
      setLoading(urlBtn, false);
    }
  }

  async function handleAsk() {
    const question = (questionInput.value || "").trim();
    if (question.length < 2) {
      return;
    }

    appendMessage("user", question);
    questionInput.value = "";

    setLoading(askBtn, true);
    appendMessage("bot", "Thinking...");

    try {
      const resp = await fetch("/api/chat/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": csrf,
        },
        body: JSON.stringify({ question }),
      });
      const data = await resp.json();

      const pending = chatLog.querySelector(".msg.bot:last-child");
      if (pending && pending.textContent.startsWith("Thinking...")) {
        pending.remove();
      }

      if (!resp.ok || !data.ok) {
        throw new Error(extractErrorMessage(data.error || data));
      }

      appendMessage("bot", data.data.answer, data.data.retrieval || []);
    } catch (err) {
      const pending = chatLog.querySelector(".msg.bot:last-child");
      if (pending && pending.textContent.startsWith("Thinking...")) {
        pending.remove();
      }
      const message = err instanceof Error ? err.message : String(err);
      appendMessage("bot", `Error: ${message}`);
    } finally {
      setLoading(askBtn, false);
    }
  }

  async function refreshStats() {
    renderStats({}, "loading");
    try {
      const resp = await fetch("/api/stats/");
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(extractErrorMessage(data.error || data));
      }
      renderStats(data.data);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      renderStats(`Failed to load stats: ${message}`, "error");
    }
  }

  uploadBtn?.addEventListener("click", handleUpload);
  urlBtn?.addEventListener("click", handleUrlIngest);
  askBtn?.addEventListener("click", handleAsk);
  statsBtn?.addEventListener("click", refreshStats);
  pdfInput?.addEventListener("change", updatePdfSelection);

  questionInput?.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      handleAsk();
    }
  });

  initButtonAnimations();
  initFloatingPanels();
  initCursorEffects();
  updatePdfSelection();
  refreshStats();
})();
