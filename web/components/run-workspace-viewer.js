"use client";

import { useMemo } from "react";

import { TrajectoryProjection } from "@/components/chart-kit";

function publicApiUrl(path) {
  if (!path) {
    return "";
  }
  const value = String(path);
  if (value.startsWith("http://") || value.startsWith("https://") || value.startsWith("data:")) {
    return value;
  }
  const base = process.env.NEXT_PUBLIC_VLALAB_API_BASE_URL || "";
  return `${base.replace(/\/$/, "")}${value.startsWith("/") ? value : `/${value}`}`;
}

function callMaybe(fn, ...args) {
  if (typeof fn === "function") {
    fn(...args);
  }
}

function imageUrl(image = {}) {
  return publicApiUrl(image.url || image.overlay_url || image.path || "");
}

function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Number(value).toFixed(Number(value) >= 100 ? 0 : 1)} ms`;
}

function totalLatency(timing = {}) {
  if (timing.total_latency_ms !== null && timing.total_latency_ms !== undefined) {
    return timing.total_latency_ms;
  }
  const parts = [timing.transport_latency_ms, timing.inference_latency_ms].filter(
    (value) => value !== null && value !== undefined && !Number.isNaN(Number(value))
  );
  return parts.length ? parts.reduce((sum, value) => sum + Number(value), 0) : null;
}

function timingParts(timing = {}) {
  const transport = timing.transport_latency_ms ?? null;
  const inference = timing.inference_latency_ms ?? null;
  const total = totalLatency(timing);
  const gap = total !== null && transport !== null && inference !== null ? Math.max(total - transport - inference, 0) : null;
  return [
    { key: "transport", label: "Transport", value: transport },
    { key: "inference", label: "Inference", value: inference },
    { key: "gap", label: "Wait", value: gap },
  ].filter((item) => item.value !== null && item.value !== undefined && Number(item.value) > 0);
}

function vectorPreview(values = [], max = 6) {
  if (!values?.length) {
    return "--";
  }
  return values
    .slice(0, max)
    .map((value) => (value === null || value === undefined || Number.isNaN(Number(value)) ? "--" : Number(value).toFixed(3)))
    .join("  ");
}

function finiteSignalValues(series = []) {
  return (series || [])
    .flatMap((item) => item.values || [])
    .map((value) => (value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value)))
    .filter((value) => value !== null);
}

function signalBounds(series = []) {
  const values = finiteSignalValues(series);
  if (!values.length) {
    return { min: 0, max: 1 };
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    return { min: min - 1, max: max + 1 };
  }
  return { min, max };
}

function signalPoints(values = [], width, height, padding, min, max) {
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;
  const span = max - min || 1;

  return (values || [])
    .map((value, index) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return null;
      }
      const x = padding + (usableWidth * index) / Math.max(values.length - 1, 1);
      const y = height - padding - ((Number(value) - min) / span) * usableHeight;
      return `${x},${y}`;
    })
    .filter(Boolean)
    .join(" ");
}

function SignalPanel({ panel }) {
  const width = 360;
  const height = 96;
  const padding = 14;
  const series = panel?.series || [];
  const maxLength = Math.max(0, ...series.map((item) => item.values?.length || 0));
  const { min, max } = signalBounds(series);
  const markerIndex = Math.max(0, Math.min(panel?.markerIndex ?? 0, Math.max(0, maxLength - 1)));
  const markerX =
    maxLength <= 1
      ? null
      : padding + ((width - padding * 2) * markerIndex) / Math.max(maxLength - 1, 1);

  return (
    <article className="run-workspace-signal-panel">
      <div className="run-workspace-signal-header">
        <strong>{panel?.title || "Signal"}</strong>
        <span>{panel?.unit || "value"}</span>
      </div>
      {maxLength ? (
        <svg className="run-workspace-signal-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={panel?.title || "Signal"}>
          {[0, 1, 2].map((tick) => {
            const y = padding + ((height - padding * 2) * tick) / 2;
            return <line key={tick} x1={padding} y1={y} x2={width - padding} y2={y} className="run-workspace-signal-grid-line" />;
          })}
          {series.map((item) => (
            <polyline
              key={item.name}
              className="run-workspace-signal-line"
              fill="none"
              points={signalPoints(item.values || [], width, height, padding, min, max)}
              stroke={item.color || "#0f766e"}
            />
          ))}
          {markerX !== null ? (
            <line x1={markerX} y1={padding} x2={markerX} y2={height - padding} className="run-workspace-signal-marker" />
          ) : null}
        </svg>
      ) : (
        <div className="run-workspace-empty">No signal data.</div>
      )}
      <div className="run-workspace-signal-legend">
        {series.map((item) => (
          <span key={item.name}>
            <span className="run-workspace-signal-dot" style={{ backgroundColor: item.color || "#0f766e" }} />
            {item.name}
          </span>
        ))}
      </div>
    </article>
  );
}

function SignalStrip({ panels = [] }) {
  if (!panels?.length) {
    return null;
  }

  return (
    <section className="run-workspace-panel run-workspace-signal-strip">
      <div className="run-workspace-signal-grid">
        {panels.map((panel) => (
          <SignalPanel key={panel.title} panel={panel} />
        ))}
      </div>
    </section>
  );
}

function cameraOptions(images = [], allCameras = []) {
  return Array.from(new Set(["all", ...allCameras, ...images.map((image) => image.camera_name).filter(Boolean)]));
}

function normalizeTrajectory(trajectory, currentStep) {
  if (!trajectory) {
    return null;
  }
  if (Array.isArray(trajectory)) {
    return { series: trajectory, currentPoint: currentStep?.state || null, title: "Trajectory" };
  }
  return {
    title: trajectory.title || "Trajectory",
    series: trajectory.series || [],
    currentPoint: trajectory.currentPoint || currentStep?.state || null,
  };
}

function ImageGallery({ images = [], selectedCamera = "all", onCameraChange, allCameras = [], title, frameStatus = null }) {
  const cameras = cameraOptions(images, allCameras);
  const visibleImages = selectedCamera === "all" ? images : images.filter((image) => image.camera_name === selectedCamera);
  const emptyMessage =
    frameStatus?.state === "loading"
      ? `${frameStatus.label || "Loading exact frame"}...`
      : frameStatus?.state === "missing"
        ? frameStatus.label || "No images recorded for step"
        : "No images for this step.";

  return (
    <section className="run-workspace-panel run-workspace-image-panel">
      <div className="run-workspace-panel-heading">
        <div>
          <p className="run-workspace-eyebrow">Vision</p>
          <h2>{title}</h2>
        </div>
      </div>
      <div className="run-workspace-camera-strip">
        {cameras.map((camera) => (
          <button
            key={camera}
            type="button"
            className={`run-workspace-camera-button${selectedCamera === camera ? " run-workspace-is-active" : ""}`}
            onClick={() => onCameraChange?.(camera)}
          >
            {camera === "all" ? "All cameras" : camera}
          </button>
        ))}
      </div>
      <div className="run-workspace-gallery">
        {visibleImages.map((image) => (
          <figure key={`${image.camera_name}-${image.path || image.url || image.overlay_url}`} className="run-workspace-frame">
            {imageUrl(image) ? <img src={imageUrl(image)} alt={image.camera_name || "camera frame"} /> : null}
            <figcaption>{image.camera_name || "camera"}</figcaption>
          </figure>
        ))}
      </div>
      {!visibleImages.length ? <div className={`run-workspace-empty run-workspace-empty-${frameStatus?.state || "missing"}`}>{emptyMessage}</div> : null}
    </section>
  );
}

function RunWorkspaceStepDataDock({ currentStep = {}, frameStatus = null }) {
  const timing = currentStep?.timing || {};
  const actionChunk = currentStep?.action_chunk || [];
  const state = currentStep?.state || [];
  const action = currentStep?.action_preview || actionChunk?.[0] || [];

  return (
    <section className="run-workspace-panel run-workspace-data-dock">
      <article>
        <span>Current State</span>
        <strong>{vectorPreview(state, 4)}</strong>
        <p>{state.length || 0} dims</p>
      </article>
      <article>
        <span>First Action</span>
        <strong>{vectorPreview(action, 4)}</strong>
        <p>{action.length || 0} dims · {actionChunk.length || 0} chunk rows</p>
      </article>
      <article>
        <span>Latency</span>
        <strong>{formatMs(totalLatency(timing))}</strong>
        <p>Transport {formatMs(timing.transport_latency_ms)} · Inference {formatMs(timing.inference_latency_ms)}</p>
      </article>
      <article>
        <span>Frame</span>
        <strong>{frameStatus?.label || "Frame ready"}</strong>
        <p>{frameStatus?.detail || "Exact frame data is loaded."}</p>
      </article>
    </section>
  );
}

function InlinePlaybackControls({
  stepIdx = 0,
  maxStep = 0,
  isPlaying = false,
  playbackMs = 220,
  playbackOptions = [],
  stepInput = "0",
  onStepChange,
  onPlayToggle,
  onPlaybackMsChange,
  onStepInputChange,
  onStepInputCommit,
}) {
  const options = playbackOptions.length
    ? playbackOptions
    : [
        { value: 120, label: "Fast" },
        { value: 220, label: "Normal" },
        { value: 420, label: "Slow" },
      ];

  return (
    <section className="run-workspace-inline-playback" aria-label="Replay controls">
      <div className="run-workspace-inline-buttons">
        <button type="button" onClick={() => onStepChange?.(0)} title="First step">|&lt;</button>
        <button type="button" onClick={() => onStepChange?.(stepIdx - 1)} title="Previous step">&lt;</button>
        <button type="button" className="run-workspace-inline-play-button" onClick={() => onPlayToggle?.()}>
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={() => onStepChange?.(stepIdx + 1)} title="Next step">&gt;</button>
        <button type="button" onClick={() => onStepChange?.(maxStep)} title="Last step">&gt;|</button>
      </div>

      <input
        className="run-workspace-inline-slider"
        type="range"
        min="0"
        max={maxStep}
        value={stepIdx}
        onChange={(event) => onStepChange?.(Number(event.target.value))}
      />

      <div className="run-workspace-inline-meta">
        <strong className="run-workspace-inline-readout">{stepIdx} / {maxStep}</strong>
        <div className="run-workspace-inline-options">
          <label>
            <span>Step</span>
            <input
              type="number"
              min="0"
              max={maxStep}
              value={stepInput}
              onChange={(event) => onStepInputChange?.(event.target.value)}
              onBlur={() => onStepInputCommit?.(stepInput)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  onStepInputCommit?.(event.currentTarget.value);
                }
              }}
            />
          </label>
          <label>
            <span>Speed</span>
            <select value={playbackMs} onChange={(event) => onPlaybackMsChange?.(Number(event.target.value))}>
              {options.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
        </div>
      </div>
    </section>
  );
}

function AttentionOverlayGrid({ sourceImages = [], overlays = [] }) {
  const sourceByCamera = new Map(sourceImages.map((image) => [image.camera_name, image]));
  const overlayByCamera = new Map(overlays.map((image) => [image.camera_name, image]));
  const cameras = Array.from(new Set([...sourceByCamera.keys(), ...overlayByCamera.keys()]));

  if (!cameras.length) {
    return <div className="run-workspace-empty">No attention overlays for this step.</div>;
  }

  return (
    <div className="run-workspace-attention-grid">
      {cameras.map((camera) => {
        const source = sourceByCamera.get(camera);
        const overlay = overlayByCamera.get(camera);
        return (
          <article key={camera} className="run-workspace-attention-card">
            <h3>{camera}</h3>
            <div className="run-workspace-attention-pair">
              <figure>
                {source ? <img src={imageUrl(source)} alt={`${camera} source`} /> : <span>No source</span>}
                <figcaption>Source</figcaption>
              </figure>
              <figure>
                {overlay ? (
                  <img
                    src={publicApiUrl(overlay.overlay_url || overlay.url || overlay.path || "")}
                    alt={`${camera} attention overlay`}
                  />
                ) : (
                  <span>No overlay</span>
                )}
                <figcaption>Overlay</figcaption>
              </figure>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function RunWorkspaceAttentionWorkbench({
  currentStep = {},
  images = [],
  attentionData = null,
  attentionLoading = false,
  attentionControls = {},
  stepIdx = 0,
}) {
  const stdoutTail = attentionData?.stdout_tail || attentionControls?.stdoutTail || "";
  const hasOverlay = Boolean(attentionData?.overlays?.length);
  const status = attentionLoading ? "Generating" : hasOverlay ? "Ready" : "No overlay";

  return (
    <section className="run-workspace-attention-workbench">
      <div className="run-workspace-attention-control-panel">
        <div className="run-workspace-panel-heading">
          <div>
            <p className="run-workspace-eyebrow">Attention</p>
            <h2>Source and overlay</h2>
          </div>
          <span className="run-workspace-status-pill">{status}</span>
        </div>

        <div className="run-workspace-attention-status-grid">
          <article>
            <span>Backend</span>
            <strong>{attentionData?.backend_name || "--"}</strong>
          </article>
          <article>
            <span>Overlay</span>
            <strong>{hasOverlay ? "Ready" : "Missing"}</strong>
          </article>
          <article>
            <span>Layer</span>
            <strong>{attentionControls?.layer ?? -1}</strong>
          </article>
        </div>

        <div className="run-workspace-control-grid run-workspace-attention-inputs">
          <label>
            <span>Device</span>
            <input
              value={attentionControls?.device ?? ""}
              onChange={(event) => callMaybe(attentionControls?.onDeviceChange, event.target.value)}
              placeholder="cuda:0"
            />
          </label>
          <label>
            <span>Layer</span>
            <input
              type="number"
              value={attentionControls?.layer ?? -1}
              onChange={(event) => callMaybe(attentionControls?.onLayerChange, Number(event.target.value))}
            />
          </label>
          <label>
            <span>Model override</span>
            <input
              value={attentionControls?.modelPath ?? ""}
              onChange={(event) => callMaybe(attentionControls?.onModelPathChange, event.target.value)}
              placeholder="optional path"
            />
          </label>
          <label>
            <span>Prompt override</span>
            <input
              value={attentionControls?.prompt ?? ""}
              onChange={(event) => callMaybe(attentionControls?.onPromptChange, event.target.value)}
              placeholder={currentStep?.prompt || "optional prompt"}
            />
          </label>
        </div>

        <div className="run-workspace-control-row">
          <button
            type="button"
            className="run-workspace-button"
            disabled={Boolean(attentionControls?.isGenerating)}
            onClick={() => callMaybe(attentionControls?.onGenerateCurrent, stepIdx)}
          >
            {attentionControls?.isGenerating ? "Generating" : "Generate Step"}
          </button>
        </div>

        <div className="run-workspace-attention-meta">
          <article>
            <span>Model</span>
            <p>{attentionData?.model_path || "--"}</p>
          </article>
          <article>
            <span>Output</span>
            <p>{attentionData?.output_dir || "--"}</p>
          </article>
        </div>
        {stdoutTail ? <pre className="run-workspace-log">{stdoutTail}</pre> : null}
      </div>

      <section className="run-workspace-panel run-workspace-attention-panel">
        <AttentionOverlayGrid sourceImages={images} overlays={attentionData?.overlays || []} />
      </section>
    </section>
  );
}

function TimingBreakdown({ timing = {} }) {
  const parts = timingParts(timing);
  const total = totalLatency(timing);

  if (!parts.length) {
    return <div className="run-workspace-empty">No timing breakdown.</div>;
  }

  return (
    <div className="run-workspace-timing-breakdown">
      <div className="run-workspace-timing-bar">
        {parts.map((part) => (
          <span
            key={part.key}
            className={`run-workspace-timing-segment run-workspace-timing-${part.key}`}
            style={{ flexGrow: Math.max(Number(part.value), 1) }}
            title={`${part.label}: ${formatMs(part.value)}`}
          />
        ))}
      </div>
      <div className="run-workspace-timing-legend">
        {parts.map((part) => (
          <span key={part.key}>
            {part.label} {formatMs(part.value)}
          </span>
        ))}
        <strong>Total {formatMs(total)}</strong>
      </div>
    </div>
  );
}

export function RunWorkspaceViewer({
  mode = "replay",
  currentStep = {},
  selectedCamera = "all",
  onCameraChange,
  allCameras = [],
  stepIdx = 0,
  maxStep = 0,
  isPlaying = false,
  playbackMs = 220,
  playbackOptions = [],
  stepInput = "0",
  onStepChange,
  onPlayToggle,
  onPlaybackMsChange,
  onStepInputChange,
  onStepInputCommit,
  frameStatus = null,
  attentionData = null,
  attentionLoading = false,
  attentionControls = {},
  trajectory = null,
  signalPanels = [],
  showTrajectory = true,
  children,
}) {
  const normalizedTrajectory = useMemo(() => normalizeTrajectory(trajectory, currentStep), [trajectory, currentStep]);
  const images = currentStep?.images || [];

  return (
    <main className={`run-workspace-viewer run-workspace-viewer-${mode}`}>
      {mode === "replay" ? (
        <section className="run-workspace-visual-stack">
          <InlinePlaybackControls
            stepIdx={stepIdx}
            maxStep={maxStep}
            isPlaying={isPlaying}
            playbackMs={playbackMs}
            playbackOptions={playbackOptions}
            stepInput={stepInput}
            onStepChange={onStepChange}
            onPlayToggle={onPlayToggle}
            onPlaybackMsChange={onPlaybackMsChange}
            onStepInputChange={onStepInputChange}
            onStepInputCommit={onStepInputCommit}
          />
          <SignalStrip panels={signalPanels} />
          <ImageGallery
            images={images}
            selectedCamera={selectedCamera}
            onCameraChange={onCameraChange}
            allCameras={allCameras}
            title="Observation"
            frameStatus={frameStatus}
          />
          <RunWorkspaceStepDataDock currentStep={currentStep} frameStatus={frameStatus} />
        </section>
      ) : mode === "attention" ? (
        <RunWorkspaceAttentionWorkbench
          currentStep={currentStep}
          images={images}
          attentionData={attentionData}
          attentionLoading={attentionLoading}
          attentionControls={attentionControls}
          stepIdx={stepIdx}
        />
      ) : null}

      {mode === "analyze" && children ? <section className="run-workspace-analyze-slot">{children}</section> : null}

      {showTrajectory && normalizedTrajectory?.series?.length ? (
        <section className="run-workspace-panel run-workspace-trajectory-panel">
          <div className="run-workspace-panel-heading">
            <div>
              <p className="run-workspace-eyebrow">Trajectory</p>
              <h2>State and prediction</h2>
            </div>
          </div>
          <TrajectoryProjection
            title={normalizedTrajectory.title}
            series={normalizedTrajectory.series}
            currentPoint={normalizedTrajectory.currentPoint}
          />
        </section>
      ) : null}

      {mode === "attention" ? (
        <section className="run-workspace-panel run-workspace-step-summary">
          <div className="run-workspace-summary-grid">
            <article>
              <span>Prompt</span>
              <p>{currentStep?.prompt || "--"}</p>
            </article>
            <article>
              <span>State</span>
              <p>{vectorPreview(currentStep?.state || [])}</p>
            </article>
            <article>
              <span>First action</span>
              <p>{vectorPreview(currentStep?.action_preview || [])}</p>
            </article>
            <article>
              <span>Timing</span>
              <TimingBreakdown timing={currentStep?.timing || {}} />
            </article>
          </div>
        </section>
      ) : null}
    </main>
  );
}
