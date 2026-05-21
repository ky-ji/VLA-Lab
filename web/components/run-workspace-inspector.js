"use client";

function isFiniteNumber(value) {
  return value !== null && value !== undefined && !Number.isNaN(Number(value));
}

function formatNumber(value, digits = 3) {
  if (!isFiniteNumber(value)) {
    return "--";
  }
  const number = Number(value);
  if (Math.abs(number) >= 1000 || Math.abs(number) < 0.001) {
    return number.toExponential(2);
  }
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  });
}

function formatMs(value) {
  if (!isFiniteNumber(value)) {
    return "--";
  }
  return `${formatNumber(value, 1)} ms`;
}

function formatText(value, fallback = "--") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  return String(value);
}

function vectorRows(values = [], labels = [], limit = 12) {
  return (values || []).slice(0, limit).map((value, idx) => ({
    key: labels?.[idx] || `dim_${idx}`,
    value,
  }));
}

function timingTotal(timing = {}) {
  if (isFiniteNumber(timing.total_latency_ms)) {
    return Number(timing.total_latency_ms);
  }
  const parts = [timing.transport_latency_ms, timing.inference_latency_ms].filter(isFiniteNumber);
  if (!parts.length) {
    return null;
  }
  return parts.reduce((sum, value) => sum + Number(value), 0);
}

function latencyBreakdown(timing = {}) {
  const total = timingTotal(timing);
  const transport = isFiniteNumber(timing.transport_latency_ms) ? Number(timing.transport_latency_ms) : null;
  const inference = isFiniteNumber(timing.inference_latency_ms) ? Number(timing.inference_latency_ms) : null;
  const message = isFiniteNumber(timing.message_interval_ms) ? Number(timing.message_interval_ms) : null;
  const accounted = [transport, inference].filter(isFiniteNumber).reduce((sum, value) => sum + Number(value), 0);
  const waiting = isFiniteNumber(total) && accounted > 0 ? Math.max(Number(total) - accounted, 0) : null;

  return [
    { key: "transport", label: "Transport", value: transport },
    { key: "inference", label: "Inference", value: inference },
    { key: "waiting", label: "Other wait", value: waiting },
    { key: "message", label: "Message gap", value: message },
  ].filter((item) => isFiniteNumber(item.value));
}

function compactActionChunkSummary(chunk = [], labels = []) {
  if (!chunk?.length) {
    return { rows: 0, dims: 0, dominant: "--", span: null, first: [], last: [] };
  }

  const dims = Math.max(0, ...(chunk || []).map((row) => row?.length || 0));
  const spans = Array.from({ length: dims }, (_, dim) => {
    const values = chunk.map((row) => row?.[dim]).filter(isFiniteNumber).map(Number);
    if (!values.length) {
      return { dim, span: null };
    }
    return { dim, span: Math.max(...values) - Math.min(...values) };
  });
  const dominant = spans
    .filter((item) => isFiniteNumber(item.span))
    .reduce((best, item) => (Number(item.span) > Number(best.span) ? item : best), { dim: -1, span: null });

  return {
    rows: chunk.length,
    dims,
    dominant: dominant.dim >= 0 ? labels?.[dominant.dim] || `dim_${dominant.dim}` : "--",
    span: dominant.span,
    first: chunk[0] || [],
    last: chunk[chunk.length - 1] || [],
  };
}

function safeEntries(value = {}) {
  return Object.entries(value || {}).filter(([, item]) => item !== null && item !== undefined && item !== "");
}

function callMaybe(fn, ...args) {
  if (typeof fn === "function") {
    fn(...args);
  }
}

function JsonDetails({ title = "Raw JSON", value }) {
  return (
    <details className="run-workspace-details">
      <summary className="run-workspace-details-summary">{title}</summary>
      <pre className="run-workspace-json">{JSON.stringify(value ?? null, null, 2)}</pre>
    </details>
  );
}

function StatCard({ label, value, note }) {
  return (
    <article className="run-workspace-stat-card">
      <span className="run-workspace-stat-label">{label}</span>
      <strong className="run-workspace-stat-value">{value}</strong>
      {note ? <span className="run-workspace-stat-note">{note}</span> : null}
    </article>
  );
}

function VectorList({ title, values = [], labels = [] }) {
  const rows = vectorRows(values, labels);
  return (
    <section className="run-workspace-panel">
      <div className="run-workspace-panel-header">
        <h3 className="run-workspace-panel-title">{title}</h3>
        <span className="run-workspace-pill">{values?.length || 0} dims</span>
      </div>
      {rows.length ? (
        <div className="run-workspace-vector-list">
          {rows.map((row) => (
            <div key={row.key} className="run-workspace-vector-row">
              <span className="run-workspace-vector-key">{row.key}</span>
              <span className="run-workspace-vector-value">{formatNumber(row.value, 4)}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="run-workspace-empty">No vector data.</p>
      )}
    </section>
  );
}

function LatencyPanel({ currentStep, latencySummary }) {
  const timing = currentStep?.timing || {};
  const parts = latencyBreakdown(timing);
  const total = timingTotal(timing);
  const p95 = latencySummary?.total_latency?.p95_ms ?? latencySummary?.p95_ms;

  return (
    <section className="run-workspace-panel">
      <div className="run-workspace-panel-header">
        <h3 className="run-workspace-panel-title">Latency</h3>
        <span className="run-workspace-pill">{formatMs(total)}</span>
      </div>
      <div className="run-workspace-latency-stack">
        {parts.length ? (
          parts.map((part) => (
            <div key={part.key} className="run-workspace-latency-row">
              <span className="run-workspace-latency-label">{part.label}</span>
              <span className="run-workspace-latency-track">
                <span
                  className={`run-workspace-latency-fill run-workspace-latency-fill-${part.key}`}
                  style={{ width: `${Math.min(100, Math.max(4, (Number(part.value) / Math.max(Number(total) || Number(part.value), 1)) * 100))}%` }}
                />
              </span>
              <span className="run-workspace-latency-value">{formatMs(part.value)}</span>
            </div>
          ))
        ) : (
          <p className="run-workspace-empty">No latency breakdown for this step.</p>
        )}
      </div>
      {isFiniteNumber(p95) ? <p className="run-workspace-note">Run P95 total latency: {formatMs(p95)}</p> : null}
    </section>
  );
}

function ActionChunkPanel({ currentStep, replay, currentBoundary, currentExecution }) {
  const labels = replay?.action_labels || replay?.expanded_execution?.action_labels || [];
  const chunk = currentStep?.action_chunk || [];
  const summary = compactActionChunkSummary(chunk, labels);
  const executionStart = currentBoundary?.start;
  const executionEnd = currentBoundary?.end;

  return (
    <section className="run-workspace-panel">
      <div className="run-workspace-panel-header">
        <h3 className="run-workspace-panel-title">Action Chunk</h3>
        <span className="run-workspace-pill">{summary.rows} x {summary.dims}</span>
      </div>
      <div className="run-workspace-stat-grid">
        <StatCard label="Rows" value={summary.rows} note="planned actions" />
        <StatCard label="Dominant" value={summary.dominant} note={`span ${formatNumber(summary.span, 3)}`} />
        <StatCard
          label="Execution"
          value={isFiniteNumber(executionStart) && isFiniteNumber(executionEnd) ? `${executionStart}-${Math.max(executionStart, executionEnd - 1)}` : "--"}
          note={`${currentExecution?.length || 0} expanded rows`}
        />
      </div>
      <div className="run-workspace-two-column">
        <VectorList title="First Action" values={summary.first} labels={labels} />
        <VectorList title="Last Action" values={summary.last} labels={labels} />
      </div>
    </section>
  );
}

function ModeButton({ mode, currentMode, onModeChange, label }) {
  const active = mode === currentMode;
  return (
    <button
      type="button"
      className={`run-workspace-mode-button${active ? " run-workspace-mode-button-active" : ""}`}
      aria-pressed={active}
      onClick={() => callMaybe(onModeChange, mode)}
    >
      {label}
    </button>
  );
}

export function RunWorkspaceRail({
  mode,
  onModeChange,
  collapsed,
  onToggleCollapsed,
  summary = {},
  currentStep = {},
  stepIdx = 0,
  maxStep = 0,
  frameStatus = null,
  allCameras = [],
  selectedCamera,
  onCameraChange,
  rerunStatus,
}) {
  const modes = [
    ["replay", "Replay"],
    ["attention", "Attention"],
    ["analyze", "Analyze"],
  ];
  const statusText = rerunStatus?.available && !rerunStatus?.stale ? "Ready" : rerunStatus?.available ? "Stale" : "Missing";
  const actionSummary = compactActionChunkSummary(currentStep?.action_chunk || [], []);
  const latency = timingTotal(currentStep?.timing || {});
  const imageCount = currentStep?.images?.length || 0;

  return (
    <aside className={`run-workspace-rail${collapsed ? " run-workspace-rail-collapsed" : ""}`}>
      <div className="run-workspace-rail-header">
        <span className="run-workspace-rail-title">Workspace</span>
        <button type="button" className="run-workspace-icon-button" onClick={() => callMaybe(onToggleCollapsed)} aria-label="Toggle rail">
          {collapsed ? ">" : "<"}
        </button>
      </div>

      <nav className="run-workspace-mode-list" aria-label="Run workspace modes">
        {modes.map(([key, label]) => (
          <ModeButton key={key} mode={key} currentMode={mode} onModeChange={onModeChange} label={collapsed ? label.slice(0, 1) : label} />
        ))}
      </nav>

      <section className="run-workspace-rail-section">
        <h3 className="run-workspace-rail-section-title">Step Summary</h3>
        <div className="run-workspace-rail-grid">
          <article>
            <span>Step</span>
            <strong>{stepIdx} / {maxStep}</strong>
          </article>
          <article>
            <span>Frame</span>
            <strong>{frameStatus?.state || "ready"}</strong>
          </article>
          <article>
            <span>Images</span>
            <strong>{imageCount}</strong>
          </article>
          <article>
            <span>Latency</span>
            <strong>{formatMs(latency)}</strong>
          </article>
        </div>
      </section>

      <section className="run-workspace-rail-section">
        <h3 className="run-workspace-rail-section-title">Run Summary</h3>
        <div className="run-workspace-rail-grid">
          <article>
            <span>Steps</span>
            <strong>{summary?.total_steps ?? maxStep + 1}</strong>
          </article>
          <article>
            <span>Action</span>
            <strong>{summary?.action_dim ?? actionSummary.dims ?? "--"}</strong>
          </article>
          <article>
            <span>Chunk</span>
            <strong>{actionSummary.rows}</strong>
          </article>
          <article>
            <span>Hz</span>
            <strong>{summary?.inference_freq ?? "--"}</strong>
          </article>
        </div>
      </section>

      <section className="run-workspace-rail-section">
        <h3 className="run-workspace-rail-section-title">Camera</h3>
        <select
          className="run-workspace-select"
          value={selectedCamera || "all"}
          onChange={(event) => callMaybe(onCameraChange, event.target.value)}
        >
          <option value="all">All cameras</option>
          {(allCameras || []).map((camera) => (
            <option key={camera} value={camera}>{camera}</option>
          ))}
        </select>
      </section>

      <section className="run-workspace-rail-section">
        <div className="run-workspace-rail-metric">
          <span className="run-workspace-stat-label">Rerun</span>
          <strong className="run-workspace-stat-value">{statusText}</strong>
        </div>
      </section>
    </aside>
  );
}

export function RunWorkspaceInspector({
  mode,
  summary = {},
  meta = {},
  replay = {},
  currentStep = {},
  stepIdx = 0,
  currentBoundary = {},
  currentExecution = [],
  latencySummary = {},
}) {
  const actionSummary = compactActionChunkSummary(currentStep?.action_chunk || [], replay?.action_labels || []);
  const tags = safeEntries(currentStep?.tags || {});

  return (
    <aside className="run-workspace-inspector">
      <header className="run-workspace-inspector-header">
        <div className="run-workspace-heading-block">
          <span className="run-workspace-eyebrow">{formatText(mode, "workspace")}</span>
          <h2 className="run-workspace-title">Step {stepIdx}</h2>
        </div>
        <span className="run-workspace-pill">{formatText(summary?.run_name || summary?.task_name, "Run")}</span>
      </header>

      <section className="run-workspace-panel">
        <div className="run-workspace-panel-header">
          <h3 className="run-workspace-panel-title">Step Context</h3>
          <span className="run-workspace-pill">{formatText(meta?.model_name || summary?.model_name, "model")}</span>
        </div>
        <p className="run-workspace-prompt">{formatText(currentStep?.prompt || meta?.task_prompt || summary?.task_name)}</p>
        {tags.length ? (
          <div className="run-workspace-token-list">
            {tags.map(([key, value]) => (
              <span key={key} className="run-workspace-token-static">{key}: {formatText(value)}</span>
            ))}
          </div>
        ) : null}
      </section>

      {mode === "replay" ? (
        <ActionChunkPanel
          currentStep={currentStep}
          replay={replay}
          currentBoundary={currentBoundary}
          currentExecution={currentExecution}
        />
      ) : null}

      <JsonDetails title="Current Step JSON" value={currentStep?.raw_step ?? currentStep} />
    </aside>
  );
}
