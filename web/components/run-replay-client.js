"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

import { browserDelete, browserFetchJson, browserPostJson, toPublicApiUrl } from "@/lib/api";
import { formatDate, formatMs, formatNumber, formatShortText } from "@/lib/format";
import {
  HeatmapChart,
  HistogramChart,
  LineChart,
  SimpleTabs,
  TimelineEvents,
  TrajectoryProjection,
} from "@/components/chart-kit";

const PALETTE = ["#0f766e", "#c2410c", "#2563eb", "#7c3aed", "#0891b2", "#be123c", "#65a30d"];

function pickColor(index) {
  return PALETTE[index % PALETTE.length];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function vectorText(values, labels, take = 8) {
  if (!values || values.length === 0) {
    return "--";
  }
  return values
    .slice(0, take)
    .map((value, idx) => `${labels?.[idx] || `dim_${idx}`}:${formatNumber(value, 3)}`)
    .join("  ");
}

function seriesForGroup(matrix, labels, group) {
  return (group?.indices || []).map((idx, offset) => ({
    name: labels?.[idx] || `dim_${idx}`,
    values: matrix.map((row) => row?.[idx] ?? null),
    color: pickColor(offset),
  }));
}

function chunkHeatmap(step) {
  const chunk = step?.action_chunk || [];
  if (!chunk.length) {
    return [];
  }
  const dimCount = chunk[0]?.length || 0;
  return Array.from({ length: dimCount }, (_, dimIdx) => chunk.map((row) => row?.[dimIdx] ?? null));
}

function positionMagnitude(actions) {
  return (actions || []).map((row) => {
    if (!row || row.length < 3) {
      return null;
    }
    return Math.sqrt((row[0] || 0) ** 2 + (row[1] || 0) ** 2 + (row[2] || 0) ** 2);
  });
}

function currentChunkBoundary(expandedExecution, stepIdx) {
  return expandedExecution?.chunk_boundaries?.[stepIdx] || { start: 0, end: 0 };
}

function stepWindow(stepIdx, maxStep, radius = 4) {
  const start = Math.max(0, stepIdx - radius);
  const end = Math.min(maxStep, stepIdx + radius);
  return Array.from({ length: end - start + 1 }, (_, idx) => start + idx);
}

function executionRows(replay, boundary) {
  const actions = replay.expanded_execution?.expanded_actions || [];
  const labels = replay.expanded_execution?.action_labels || replay.action_labels || [];
  const start = boundary?.start || 0;
  const end = boundary?.end || start;
  return actions.slice(start, end).map((row, index) => ({
    execStep: start + index,
    values: row,
    label: vectorText(row, labels, Math.min(8, labels.length || 8)),
  }));
}

function firstImageForStep(step) {
  return step?.images?.[0] || null;
}

function RawJson({ value }) {
  return (
    <details className="json-panel">
      <summary>原始 JSON</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

function PathText({ value = "--", compact = false }) {
  const text = value || "--";
  return (
    <p className={`path-text${compact ? " is-compact" : ""}`} title={text}>
      {text}
    </p>
  );
}

function VectorTable({ title, labels = [], values = [] }) {
  if (!values?.length) {
    return <div className="empty-panel">无向量数据</div>;
  }

  return (
    <div className="section-panel compact-panel">
      <p className="panel-title">{title}</p>
      <div className="table-shell">
        <table className="data-table compact-table">
          <thead>
            <tr>
              <th>维度</th>
              <th>数值</th>
            </tr>
          </thead>
          <tbody>
            {values.map((value, idx) => (
              <tr key={`${title}-${idx}`}>
                <td>{labels[idx] || `dim_${idx}`}</td>
                <td>{formatNumber(value, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ValueBarList({ title, labels = [], values = [] }) {
  const maxAbs = Math.max(1e-6, ...values.map((value) => Math.abs(value || 0)));

  if (!values?.length) {
    return <div className="empty-panel">无动作数据</div>;
  }

  return (
    <div className="section-panel compact-panel">
      <p className="panel-title">{title}</p>
      <div className="value-bar-list">
        {values.map((value, idx) => {
          const width = `${(Math.abs(value || 0) / maxAbs) * 100}%`;
          return (
            <div key={`${title}-${idx}`} className="value-bar-row">
              <span className="value-bar-label">{labels[idx] || `dim_${idx}`}</span>
              <span className="value-bar-track">
                <span
                  className={`value-bar-fill${(value || 0) < 0 ? " is-negative" : ""}`}
                  style={{ width }}
                />
              </span>
              <span className="value-bar-value">{formatNumber(value, 4)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StepImages({ images, selectedCamera, onCameraChange, title = "当前观测" }) {
  const cameras = useMemo(
    () => ["all", ...(images || []).map((image) => image.camera_name)],
    [images]
  );
  const visibleImages =
    selectedCamera === "all"
      ? images
      : (images || []).filter((image) => image.camera_name === selectedCamera);

  return (
    <div className="section-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">视觉</p>
          <h2>{title}</h2>
        </div>
      </div>
      <div className="camera-strip">
        {cameras.map((camera) => (
          <button
            key={camera}
            type="button"
            className={`camera-pill${selectedCamera === camera ? " is-active" : ""}`}
            onClick={() => onCameraChange(camera)}
          >
            {camera === "all" ? "全部相机" : camera}
          </button>
        ))}
      </div>
      <div className="step-gallery">
        {(visibleImages || []).map((image) => (
          <figure key={`${image.camera_name}-${image.path}`}>
            <img src={toPublicApiUrl(image.url || image.overlay_url)} alt={image.camera_name} />
            <figcaption>{image.camera_name}</figcaption>
          </figure>
        ))}
      </div>
      {!visibleImages?.length ? <div className="empty-panel">该 step 没有图像数据。</div> : null}
    </div>
  );
}

function StepFilmstrip({ steps, nearbySteps, currentStepIdx, onSelect, cameras = [], filmstripCamera, onCameraChange }) {
  function imageForStep(step) {
    if (!step?.images?.length) return null;
    if (!filmstripCamera) return step.images[0];
    return step.images.find((img) => img.camera_name === filmstripCamera) || step.images[0];
  }

  return (
    <div className="filmstrip-shell">
      {cameras.length > 1 ? (
        <div className="filmstrip-camera-bar">
          <span className="filmstrip-camera-label">胶片视角</span>
          {cameras.map((cam) => (
            <button
              key={cam}
              type="button"
              className={`camera-pill${filmstripCamera === cam ? " is-active" : ""}`}
              onClick={() => onCameraChange(cam)}
            >
              {cam}
            </button>
          ))}
        </div>
      ) : null}
      <div className="thumbnail-strip">
        {nearbySteps.map((idx) => {
          const step = steps[idx];
          const preview = imageForStep(step);
          return (
            <button
              key={`thumb-${idx}`}
              type="button"
              className={`thumbnail-card${idx === currentStepIdx ? " is-active" : ""}`}
              onClick={() => onSelect(idx)}
            >
              {preview ? (
                <img src={toPublicApiUrl(preview.url)} alt={`step-${idx}`} />
              ) : (
                <div className="thumbnail-empty">无图像</div>
              )}
              <div className="thumbnail-meta">
                <strong>步 {idx}</strong>
                <span>{formatMs(step?.timing?.total_latency_ms)}</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function AttentionCompareGrid({ sourceImages = [], overlays = [] }) {
  const sourceByCamera = new Map(sourceImages.map((image) => [image.camera_name, image]));
  const overlayByCamera = new Map(overlays.map((image) => [image.camera_name, image]));
  const cameras = Array.from(new Set([...sourceByCamera.keys(), ...overlayByCamera.keys()]));

  if (!cameras.length) {
    return <div className="empty-panel">当前 step 没有可对照的 attention 图像。</div>;
  }

  return (
    <div className="attention-compare-grid">
      {cameras.map((camera) => {
        const source = sourceByCamera.get(camera);
        const overlay = overlayByCamera.get(camera);
        return (
          <article key={camera} className="attention-compare-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">相机</p>
                <h2>{camera}</h2>
              </div>
            </div>
            <div className="attention-pair">
              <figure>
                {source ? <img src={toPublicApiUrl(source.url)} alt={`${camera}-source`} /> : <div className="thumbnail-empty">无源图</div>}
                <figcaption>原始图像</figcaption>
              </figure>
              <figure>
                {overlay ? <img src={toPublicApiUrl(overlay.overlay_url)} alt={`${camera}-overlay`} /> : <div className="thumbnail-empty">无叠加图</div>}
                <figcaption>Attention 叠加</figcaption>
              </figure>
            </div>
          </article>
        );
      })}
    </div>
  );
}

export default function RunReplayClient({ replay, project, runName }) {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState("replay");
  const [globalView, setGlobalView] = useState("inference");
  const [stepIdx, setStepIdx] = useState(0);
  const [stepInput, setStepInput] = useState("0");
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackMs, setPlaybackMs] = useState(220);
  const [selectedCamera, setSelectedCamera] = useState("all");
  const [attentionLayer, setAttentionLayer] = useState(-1);
  const [attentionDevice, setAttentionDevice] = useState("cuda:0");
  const [attentionModelPath, setAttentionModelPath] = useState("");
  const [attentionPrompt, setAttentionPrompt] = useState("");
  const [attentionData, setAttentionData] = useState(null);
  const [attentionError, setAttentionError] = useState("");
  const [attentionStdoutTail, setAttentionStdoutTail] = useState("");
  const [isLoadingAttention, setIsLoadingAttention] = useState(false);
  const [isGeneratingAttention, setIsGeneratingAttention] = useState(false);
  const [attentionReloadKey, setAttentionReloadKey] = useState(0);
  const [filmstripCamera, setFilmstripCamera] = useState("");
  const [deleteArmed, setDeleteArmed] = useState(false);
  const [isPending, startTransition] = useTransition();

  const summary = replay.summary;
  const meta = replay.meta || {};
  const steps = replay.step_details || [];
  const currentStep = steps[stepIdx] || steps[0] || {};
  const maxStep = Math.max(0, steps.length - 1);
  const currentBoundary = currentChunkBoundary(replay.expanded_execution, stepIdx);
  const nearbySteps = stepWindow(stepIdx, maxStep, 5);
  const allCameras = useMemo(() => replay.camera_names || [], [replay.camera_names]);
  const currentExecution = useMemo(() => executionRows(replay, currentBoundary), [replay, currentBoundary]);
  const playbackOptions = [
    { label: "慢", value: 500 },
    { label: "中", value: 220 },
    { label: "快", value: 120 },
  ];

  useEffect(() => {
    setStepInput(String(stepIdx));
  }, [stepIdx]);

  useEffect(() => {
    const imageNames = (currentStep.images || []).map((image) => image.camera_name);
    if (selectedCamera !== "all" && !imageNames.includes(selectedCamera)) {
      setSelectedCamera("all");
    }
  }, [currentStep.images, selectedCamera]);

  useEffect(() => {
    if (!isPlaying || maxStep <= 0) {
      return undefined;
    }
    const timer = window.setTimeout(() => {
      startTransition(() => setStepIdx((value) => (value >= maxStep ? 0 : value + 1)));
    }, playbackMs);
    return () => window.clearTimeout(timer);
  }, [isPlaying, maxStep, playbackMs, startTransition, stepIdx]);

  useEffect(() => {
    function onKeyDown(event) {
      const tag = event.target?.tagName?.toLowerCase?.();
      if (tag === "input" || tag === "textarea" || tag === "select") {
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        startTransition(() => setStepIdx((prev) => clamp(prev - 1, 0, maxStep)));
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        startTransition(() => setStepIdx((prev) => clamp(prev + 1, 0, maxStep)));
      } else if (event.key === " ") {
        event.preventDefault();
        setIsPlaying((value) => !value);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [maxStep, startTransition]);

  useEffect(() => {
    if (activeTab !== "attention") {
      return undefined;
    }
    let ignore = false;

    async function loadAttention() {
      setAttentionError("");
      setIsLoadingAttention(true);
      try {
        const data = await browserFetchJson(`/api/runs/${project}/${runName}/attention`, {
          step_idx: stepIdx,
          attention_layer: attentionLayer,
          model_path_override: attentionModelPath || undefined,
          prompt_override: attentionPrompt || undefined,
        });
        if (!ignore) {
          setAttentionData(data);
          setAttentionStdoutTail(data?.stdout_tail || "");
        }
      } catch (error) {
        if (!ignore) {
          setAttentionError(String(error.message || error));
          setAttentionData(null);
        }
      } finally {
        if (!ignore) {
          setIsLoadingAttention(false);
        }
      }
    }

    loadAttention();
    return () => {
      ignore = true;
    };
  }, [activeTab, attentionDevice, attentionLayer, attentionModelPath, attentionPrompt, project, runName, stepIdx, attentionReloadKey]);

  function jumpToStep(nextStep) {
    const clamped = clamp(Number(nextStep), 0, maxStep);
    startTransition(() => setStepIdx(clamped));
  }

  async function handleDelete() {
    await browserDelete(`/api/runs/${project}/${runName}`);
    router.push("/runs");
    router.refresh();
  }

  async function handleGenerateAttention(mode) {
    const requestedSteps =
      mode === "window"
        ? stepWindow(stepIdx, maxStep, 2)
        : [stepIdx];

    try {
      setIsGeneratingAttention(true);
      setAttentionError("");
      const data = await browserPostJson(`/api/runs/${project}/${runName}/attention/generate`, {
        requested_steps: requestedSteps,
        focus_step: stepIdx,
        device: attentionDevice,
        attention_layer: attentionLayer,
        model_path_override: attentionModelPath || undefined,
        prompt_override: attentionPrompt || undefined,
      });
      setAttentionData(data);
      setAttentionStdoutTail(data?.stdout_tail || "");
    } catch (error) {
      setAttentionError(String(error.message || error));
    } finally {
      setIsGeneratingAttention(false);
    }
  }

  const currentCachedSteps = new Set(attentionData?.cached_steps || []);

  return (
    <div className="section-stack">
      <section className="section-panel">
        <div className="run-detail-header">
          <div>
            <p className="eyebrow">{summary.project}</p>
            <h1>{summary.run_name}</h1>
            <p className="hero-desc">{meta.task_prompt || summary.task_name || "--"}</p>
          </div>
          <div className="run-meta-grid">
            <div className="run-meta-item">
              <span className="run-meta-label">模型</span>
              <span className="run-meta-value">{summary.model_name || "unknown"}</span>
            </div>
            <div className="run-meta-item">
              <span className="run-meta-label">步数</span>
              <span className="run-meta-value">{summary.total_steps ?? 0}</span>
            </div>
            {summary.inference_freq ? (
              <div className="run-meta-item">
                <span className="run-meta-label">频率</span>
                <span className="run-meta-value">{summary.inference_freq} Hz</span>
              </div>
            ) : null}
            {summary.action_dim ? (
              <div className="run-meta-item">
                <span className="run-meta-label">动作维度</span>
                <span className="run-meta-value">{summary.action_dim}</span>
              </div>
            ) : null}
            <div className="run-meta-item">
              <span className="run-meta-label">更新</span>
              <span className="run-meta-value">{formatDate(summary.updated_at)}</span>
            </div>
          </div>
        </div>
      </section>

      <section className="selection-panel">
        <div className="replay-toolbar">
          <div className="replay-row-main">
            <div className="replay-btn-group">
              <button type="button" className="replay-btn" onClick={() => jumpToStep(0)} title="首帧">⏮</button>
              <button type="button" className="replay-btn" onClick={() => jumpToStep(stepIdx - 1)} title="上一帧">◀</button>
              <button type="button" className="replay-btn replay-btn-play" onClick={() => setIsPlaying((v) => !v)}>
                {isPlaying ? "⏸" : "▶"}
              </button>
              <button type="button" className="replay-btn" onClick={() => jumpToStep(stepIdx + 1)} title="下一帧">▶</button>
              <button type="button" className="replay-btn" onClick={() => jumpToStep(maxStep)} title="末帧">⏭</button>
            </div>
            <input
              type="range"
              className="replay-slider"
              min="0"
              max={maxStep}
              value={stepIdx}
              onChange={(event) => jumpToStep(event.target.value)}
            />
            <span className="replay-readout">{stepIdx} / {maxStep}</span>
          </div>
          <div className="replay-row-options">
            <label className="replay-option">
              <span>跳转</span>
              <input
                type="number"
                min="0"
                max={maxStep}
                value={stepInput}
                onChange={(event) => setStepInput(event.target.value)}
                onBlur={() => jumpToStep(stepInput)}
              />
            </label>
            <label className="replay-option">
              <span>速度</span>
              <select value={playbackMs} onChange={(event) => setPlaybackMs(Number(event.target.value))}>
                {playbackOptions.map((item) => (
                  <option key={item.value} value={item.value}>{item.label}</option>
                ))}
              </select>
            </label>
            <span className="muted replay-hint">快捷键: ← → 切帧 · 空格播放</span>
          </div>
        </div>
        <StepFilmstrip
          steps={steps}
          nearbySteps={nearbySteps}
          currentStepIdx={stepIdx}
          onSelect={jumpToStep}
          cameras={allCameras}
          filmstripCamera={filmstripCamera || allCameras[0] || ""}
          onCameraChange={setFilmstripCamera}
        />
        <SimpleTabs
          activeTab={activeTab}
          onChange={setActiveTab}
          tabs={[
            { key: "replay", label: "逐帧回放" },
            { key: "attention", label: "Attention" },
            { key: "latency", label: "时延分析" },
            { key: "action", label: "动作分析" },
            { key: "model", label: "模型配置" },
          ]}
        />
      </section>

      {activeTab === "replay" ? (
        <>
          {/* 第一行：视觉观测 + 当前帧指标 */}
          <section className="detail-grid">
            <StepImages
              images={currentStep.images || []}
              selectedCamera={selectedCamera}
              onCameraChange={setSelectedCamera}
              title="视觉观测"
            />
            <div className="section-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">当前帧</p>
                  <h2>延迟、提示词与时间线</h2>
                </div>
                <span className="frame-badge">步 {stepIdx}</span>
              </div>

              {/* 延迟三卡片 —— 带颜色区分 */}
              <div className="latency-trio">
                <div className="latency-kpi accent-amber">
                  <span className="latency-kpi-label">传输</span>
                  <strong className="latency-kpi-value">{formatMs(currentStep.timing?.transport_latency_ms)}</strong>
                </div>
                <div className="latency-kpi accent-blue">
                  <span className="latency-kpi-label">推理</span>
                  <strong className="latency-kpi-value">{formatMs(currentStep.timing?.inference_latency_ms)}</strong>
                </div>
                <div className="latency-kpi accent-teal">
                  <span className="latency-kpi-label">总计</span>
                  <strong className="latency-kpi-value">{formatMs(currentStep.timing?.total_latency_ms)}</strong>
                </div>
              </div>

              {/* 结构化元数据 */}
              <dl className="step-meta-list">
                <div className="step-meta-row">
                  <dt>提示词</dt>
                  <dd>{currentStep.prompt || meta.task_prompt || "--"}</dd>
                </div>
                {Object.keys(currentStep.tags || {}).length ? (
                  <div className="step-meta-row">
                    <dt>标签</dt>
                    <dd className="mono-text">{JSON.stringify(currentStep.tags)}</dd>
                  </div>
                ) : null}
                <div className="step-meta-row">
                  <dt>状态快照</dt>
                  <dd className="mono-text">{vectorText(currentStep.state, replay.state_labels)}</dd>
                </div>
                <div className="step-meta-row">
                  <dt>首个动作</dt>
                  <dd className="mono-text">{vectorText(currentStep.action_preview, replay.action_labels)}</dd>
                </div>
              </dl>

              <TimelineEvents events={currentStep.timeline_events} />
            </div>
          </section>

          {/* 第二行：3D 规划 + 动作块 */}
          <section className="detail-grid">
            <div className="section-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">规划</p>
                  <h2>3D 动作轨迹</h2>
                </div>
              </div>
              <TrajectoryProjection
                title="历史轨迹 + 预测"
                currentPoint={currentStep.state}
                series={[
                  {
                    name: "历史",
                    color: "#4b5563",
                    points: replay.states.slice(0, stepIdx + 1),
                  },
                  {
                    name: "预测",
                    color: "#dc2626",
                    dashed: true,
                    points: currentStep.action_chunk,
                  },
                ]}
              />
              <section className="detail-grid" style={{ marginTop: "16px" }}>
                <VectorTable title="当前状态" labels={replay.state_labels} values={currentStep.state || []} />
                <VectorTable title="首个执行动作" labels={replay.action_labels} values={currentStep.action_preview || []} />
              </section>
            </div>

            <div className="section-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">动作块</p>
                  <h2>当前推理窗口</h2>
                </div>
                {currentStep.action_chunk?.length ? (
                  <span className="frame-badge">
                    {currentStep.action_chunk.length} 步 × {(currentStep.action_chunk[0] || []).length} 维
                  </span>
                ) : null}
              </div>

              {/* 热图全宽展示 */}
              <HeatmapChart
                matrix={chunkHeatmap(currentStep)}
                rowLabels={replay.action_labels}
                title={`步 ${stepIdx} · 动作块热图`}
              />

              {/* 首动作条形图 + 执行窗口并排 */}
              <div className="chunk-bottom-grid">
                <ValueBarList
                  title="首个执行动作"
                  labels={replay.action_labels}
                  values={currentStep.action_preview || []}
                />
                {currentExecution.length ? (
                  <div className="section-panel compact-panel">
                    <p className="panel-title">执行窗口（步 {currentBoundary.start}–{currentBoundary.end - 1}）</p>
                    <div className="table-shell">
                      <table className="data-table compact-table">
                        <thead>
                          <tr>
                            <th>执行步</th>
                            <th>动作向量</th>
                          </tr>
                        </thead>
                        <tbody>
                          {currentExecution.map((row) => (
                            <tr key={row.execStep}>
                              <td>{row.execStep}</td>
                              <td className="mono-text">{row.label}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </section>

          <section className="section-panel">
            <div className="section-heading">
              <div>
                <p className="eyebrow">全局趋势</p>
                <h2>状态与动作时序曲线</h2>
              </div>
            </div>
            <SimpleTabs
              activeTab={globalView}
              onChange={setGlobalView}
              tabs={[
                { key: "inference", label: "推理步视图" },
                { key: "execution", label: "执行步视图" },
              ]}
            />

            {globalView === "inference" ? (
              <div className="section-stack">
                {replay.state_groups?.map((group) =>
                  group.indices?.length ? (
                    <LineChart
                      key={`state-${group.key}`}
                      title={`状态｜${group.title}`}
                      markerIndex={stepIdx}
                      xLabel="X 轴: 推理调用次数"
                      series={seriesForGroup(replay.states, replay.state_labels, group)}
                    />
                  ) : null
                )}
                {replay.action_groups?.map((group) =>
                  group.indices?.length ? (
                    <LineChart
                      key={`action-${group.key}`}
                      title={`动作｜${group.title}`}
                      markerIndex={stepIdx}
                      xLabel="X 轴: 推理调用次数"
                      series={seriesForGroup(replay.first_actions, replay.action_labels, group)}
                    />
                  ) : null
                )}
              </div>
            ) : (
              <div className="section-stack">
                {replay.expanded_execution?.action_groups?.map((group) =>
                  group.indices?.length ? (
                    <LineChart
                      key={`exec-${group.key}`}
                      title={`执行动作｜${group.title}`}
                      markerIndex={currentBoundary.start}
                      xLabel="X 轴: 全局执行步"
                      series={seriesForGroup(
                        replay.expanded_execution.expanded_actions,
                        replay.expanded_execution.action_labels,
                        group
                      )}
                    />
                  ) : null
                )}
              </div>
            )}
            <RawJson value={currentStep.raw_step} />
          </section>
        </>
      ) : null}

      {activeTab === "attention" ? (
        <section className="section-stack">
          <section className="selection-panel">
            <div className="section-heading">
              <div>
                <p className="eyebrow">注意力</p>
                <h2>缓存浏览与离线生成</h2>
              </div>
            </div>
            <div className="control-grid">
              <label>
                <span>设备</span>
                <input value={attentionDevice} onChange={(event) => setAttentionDevice(event.target.value)} placeholder="cuda:0" />
              </label>
              <label>
                <span>层索引</span>
                <input type="number" value={attentionLayer} onChange={(event) => setAttentionLayer(Number(event.target.value))} />
              </label>
              <label>
                <span>覆盖模型路径</span>
                <input
                  value={attentionModelPath}
                  onChange={(event) => setAttentionModelPath(event.target.value)}
                  placeholder="可选：覆盖模型路径"
                />
              </label>
              <label>
                <span>覆盖提示词</span>
                <input
                  value={attentionPrompt}
                  onChange={(event) => setAttentionPrompt(event.target.value)}
                  placeholder="可选：覆盖提示词"
                />
              </label>
            </div>
            <div className="selection-actions">
              <button type="button" disabled={isGeneratingAttention} onClick={() => handleGenerateAttention("current")}>
                {isGeneratingAttention ? "生成中..." : "生成当前 Step"}
              </button>
              <button type="button" disabled={isGeneratingAttention} onClick={() => handleGenerateAttention("window")}>
                {isGeneratingAttention ? "请稍候" : "生成附近 5 步"}
              </button>
              <button type="button" className="secondary-button" onClick={() => setAttentionReloadKey((value) => value + 1)}>
                刷新当前状态
              </button>
              <span className="muted">
                已发现缓存: {(replay.attention_caches || []).map((item) => item.name).join(", ") || "无"} | 快捷键: 左右切帧, 空格播放
              </span>
            </div>
            {attentionError ? <div className="placeholder-note">{attentionError}</div> : null}
          </section>

          <section className="detail-grid">
            <div className="section-stack">
              <StepImages
                images={currentStep.images || []}
                selectedCamera={selectedCamera}
                onCameraChange={setSelectedCamera}
                title="当前观测图像"
              />
              <div className="section-panel">
                <div className="section-heading">
                  <div>
              <p className="eyebrow">缓存窗口</p>
                  <h2>附近步缓存状态</h2>
                  </div>
                </div>
                <div className="step-strip">
                  {nearbySteps.map((idx) => (
                    <button
                      key={`attn-${idx}`}
                      type="button"
                      className={`step-token${idx === stepIdx ? " is-active" : ""}${currentCachedSteps.has(idx) ? " is-cached" : ""}`}
                      onClick={() => jumpToStep(idx)}
                    >
                      <strong>{idx}</strong>
                      <span>{currentCachedSteps.has(idx) ? "已缓存" : "未缓存"}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="section-stack">
              <div className="timing-grid">
                <div className="kpi-card">
                  <span className="stat-label">当前 Step</span>
                  <strong>{stepIdx}</strong>
                </div>
                <div className="kpi-card">
                  <span className="stat-label">当前缓存</span>
                  <strong>{attentionData?.current_cached ? "是" : "否"}</strong>
                </div>
                <div className="kpi-card">
                  <span className="stat-label">已缓存 Step</span>
                  <strong>{attentionData?.cached_step_count ?? 0}</strong>
                </div>
                <div className="kpi-card">
                  <span className="stat-label">输出目录</span>
                  <PathText value={attentionData?.output_dir || "--"} compact />
                </div>
                <div className="kpi-card">
                  <span className="stat-label">状态</span>
                  <strong>{isGeneratingAttention ? "正在生成" : isLoadingAttention ? "读取缓存" : "就绪"}</strong>
                </div>
              </div>
              <div className="meta-panel">
                <div>
                  <span className="stat-label">实际模型路径</span>
                  <PathText value={attentionData?.model_path || meta.model_path || "--"} />
                </div>
                <div>
                  <span className="stat-label">实际提示词</span>
                  <p>{attentionData?.prompt || currentStep.prompt || meta.task_prompt || "--"}</p>
                </div>
                {attentionStdoutTail ? (
                  <div>
                    <span className="stat-label">生成日志</span>
                    <pre>{attentionStdoutTail}</pre>
                  </div>
                ) : null}
              </div>
            </div>
          </section>

          <section className="section-panel">
            <div className="section-heading">
              <div>
              <p className="eyebrow">Attention 叠加</p>
              <h2>原图 / Overlay 对照</h2>
              </div>
            </div>
            <AttentionCompareGrid sourceImages={currentStep.images || []} overlays={attentionData?.overlays || []} />
          </section>
        </section>
      ) : null}

      {activeTab === "latency" ? (
        <section className="section-stack">
          <LineChart
            title="传输 / 推理 / 总延迟趋势"
            markerIndex={stepIdx}
            xLabel="推理步"
            series={[
              { name: "传输", values: replay.timing_series.transport_latency_ms, color: "#d97706" },
              { name: "推理", values: replay.timing_series.inference_latency_ms, color: "#2563eb" },
              { name: "总计", values: replay.timing_series.total_latency_ms, color: "#6b7280" },
              { name: "消息间隔", values: replay.timing_series.message_interval_ms, color: "#0f766e" },
            ]}
          />
          <section className="detail-grid">
            <div className="timing-grid">
              {Object.entries(replay.latency_summary || {}).map(([key, value]) => {
                const labelMap = {
                  transport_latency: "传输延迟",
                  inference_latency: "推理延迟",
                  total_latency: "总延迟",
                  message_interval: "消息间隔",
                };
                return (
                  <div key={key} className="kpi-card">
                    <span className="stat-label">{labelMap[key] || key}</span>
                    <strong>{formatMs(value?.avg_ms)}</strong>
                    <span className="muted">
                      P95 {formatMs(value?.p95_ms)} · 最大 {formatMs(value?.max_ms)}
                    </span>
                  </div>
                );
              })}
            </div>
            <HistogramChart title="总延迟分布直方图" values={replay.timing_series.total_latency_ms} color="#475569" />
          </section>
        </section>
      ) : null}

      {activeTab === "action" ? (
        <section className="section-stack">
          <section className="detail-grid">
            <div className="section-stack">
              <LineChart
                title="末端位置随时间变化"
                markerIndex={stepIdx}
                xLabel="推理步"
                series={[
                  { name: "x", values: replay.first_actions.map((row) => row?.[0] ?? null), color: "#0f766e" },
                  { name: "y", values: replay.first_actions.map((row) => row?.[1] ?? null), color: "#c2410c" },
                  { name: "z", values: replay.first_actions.map((row) => row?.[2] ?? null), color: "#2563eb" },
                ]}
              />
              <LineChart
                title="动作幅度"
                markerIndex={stepIdx}
                xLabel="推理步"
                series={[{ name: "幅度", values: positionMagnitude(replay.first_actions), color: "#7c3aed" }]}
              />
            </div>
            <div className="section-stack">
              {replay.action_labels?.includes("gripper") ? (
                <LineChart
                  title="夹爪开合随时间变化"
                  markerIndex={stepIdx}
                  xLabel="推理步"
                  series={[
                    {
                      name: "夹爪",
                      values: replay.first_actions.map((row) => row?.[replay.action_labels.indexOf("gripper")] ?? null),
                      color: "#16a34a",
                    },
                  ]}
                />
              ) : null}
              <ValueBarList
                title="当前帧首动作"
                labels={replay.action_labels}
                values={currentStep.action_preview || []}
              />
            </div>
          </section>
          <section className="section-panel">
            <div className="section-heading">
              <div>
                <p className="eyebrow">统计</p>
                <h2>动作维度统计</h2>
              </div>
            </div>
            <div className="table-shell">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>维度</th>
                    <th>均值</th>
                    <th>标准差</th>
                    <th>最小值</th>
                    <th>最大值</th>
                  </tr>
                </thead>
                <tbody>
                  {(replay.action_stats || []).map((item) => (
                    <tr key={item.label}>
                      <td>{item.label}</td>
                      <td>{formatNumber(item.mean, 4)}</td>
                      <td>{formatNumber(item.std, 4)}</td>
                      <td>{formatNumber(item.min, 4)}</td>
                      <td>{formatNumber(item.max, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </section>
      ) : null}

      {activeTab === "model" ? (
        <section className="detail-grid">
          <div className="section-stack">
            <div className="section-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">配置</p>
                  <h2>部署与模型配置</h2>
                </div>
              </div>
              <div className="timing-grid">
                <div className="kpi-card">
                  <span className="stat-label">模型类型</span>
                  <strong>{meta.model_type || "未知"}</strong>
                </div>
                <div className="kpi-card">
                  <span className="stat-label">动作维度</span>
                  <strong>{summary.action_dim ?? "--"}</strong>
                </div>
                <div className="kpi-card">
                  <span className="stat-label">动作步长</span>
                  <strong>{summary.action_horizon ?? "--"}</strong>
                </div>
                <div className="kpi-card">
                  <span className="stat-label">机器人</span>
                  <strong>{meta.robot_type || summary.robot_name || "未知"}</strong>
                </div>
              </div>
              <div className="meta-panel">
                <div>
                  <span className="stat-label">模型路径</span>
                  <PathText value={meta.model_path || "--"} />
                </div>
                <div>
                  <span className="stat-label">任务提示词</span>
                  <p>{meta.task_prompt || "--"}</p>
                </div>
                <div>
                  <span className="stat-label">摄像头</span>
                  <p>{(replay.camera_names || []).join(", ") || "--"}</p>
                </div>
              </div>
            </div>
            <RawJson value={meta} />
          </div>
          <div className="section-stack">
            <div className="section-panel danger-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">危险操作</p>
                  <h2>删除运行</h2>
                </div>
              </div>
              {!deleteArmed ? (
                <button type="button" className="danger-button" onClick={() => setDeleteArmed(true)}>
                  删除此 run
                </button>
              ) : (
                <div className="selection-actions">
                  <button type="button" className="danger-button" onClick={handleDelete}>
                    确认删除
                  </button>
                  <button type="button" className="button-ghost secondary-button" onClick={() => setDeleteArmed(false)}>
                    取消
                  </button>
                </div>
              )}
            </div>
            <div className="meta-panel">
              <span className="stat-label">播放状态</span>
              <p>{isPending ? "正在切换帧..." : isPlaying ? "自动播放中" : "就绪"}</p>
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}
