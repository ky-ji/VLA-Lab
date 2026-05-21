"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useTransition } from "react";

import { browserFetchJson, browserPostJson, toPublicApiUrl } from "@/lib/api";
import { formatDate, formatMs, formatNumber } from "@/lib/format";
import {
  HistogramChart,
  LineChart,
  TimelineEvents,
} from "@/components/chart-kit";
import { RunWorkspaceInspector, RunWorkspaceRail } from "@/components/run-workspace-inspector";
import { RunWorkspaceViewer } from "@/components/run-workspace-viewer";

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

function groupByKey(groups = [], key, fallbackIndices = [], title = key) {
  return groups.find((group) => group?.key === key) || {
    key,
    title,
    indices: fallbackIndices,
  };
}

function labelIndex(labels = [], candidates = []) {
  const normalized = labels.map((label) => String(label || "").toLowerCase());
  return candidates
    .map((candidate) => normalized.indexOf(String(candidate).toLowerCase()))
    .find((idx) => idx >= 0) ?? -1;
}

function finiteNumbers(values = []) {
  return values
    .map((value) => (value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value)))
    .filter((value) => value !== null);
}

function meanValue(values = []) {
  const valid = finiteNumbers(values);
  if (!valid.length) {
    return null;
  }
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function percentileRank(values = [], target) {
  if (target === null || target === undefined || Number.isNaN(Number(target))) {
    return null;
  }
  const valid = finiteNumbers(values);
  if (!valid.length) {
    return null;
  }
  const belowOrEqual = valid.filter((value) => value <= Number(target)).length;
  return (belowOrEqual / valid.length) * 100;
}

function signedNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  const number = Number(value);
  return `${number > 0 ? "+" : ""}${number.toFixed(digits)}`;
}

function percentText(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Math.round(Number(value))}%`;
}

function totalLatency(timing = {}) {
  if (timing.total_latency_ms !== null && timing.total_latency_ms !== undefined) {
    return timing.total_latency_ms;
  }

  const parts = [timing.transport_latency_ms, timing.inference_latency_ms].filter(
    (value) => value !== null && value !== undefined && !Number.isNaN(Number(value))
  );
  if (!parts.length) {
    return null;
  }
  return parts.reduce((sum, value) => sum + Number(value), 0);
}

function timingSnapshot(timing = {}) {
  const transport = timing.transport_latency_ms ?? null;
  const inference = timing.inference_latency_ms ?? null;
  const total = totalLatency(timing);
  const messageInterval = timing.message_interval_ms ?? null;
  const gap =
    total !== null && transport !== null && inference !== null
      ? Math.max(total - transport - inference, 0)
      : null;

  return { transport, inference, total, messageInterval, gap };
}

function ratio(part, total) {
  if (
    part === null ||
    part === undefined ||
    total === null ||
    total === undefined ||
    Number(total) <= 0 ||
    Number.isNaN(Number(part)) ||
    Number.isNaN(Number(total))
  ) {
    return null;
  }
  return (Number(part) / Number(total)) * 100;
}

function latencyTone(value, summary = {}) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return { label: "No data", toneClass: "tone-muted" };
  }
  const p95 = summary?.p95_ms;
  const avg = summary?.avg_ms;
  if (p95 !== null && p95 !== undefined && Number(value) >= Number(p95)) {
    return { label: "P95+", toneClass: "tone-bad" };
  }
  if (avg !== null && avg !== undefined && Number(value) >= Number(avg) * 1.18) {
    return { label: "High", toneClass: "tone-warn" };
  }
  return { label: "Stable", toneClass: "tone-good" };
}

function latencyDriver(timing) {
  const { transport, inference, gap } = timingSnapshot(timing);
  const parts = [
    { key: "transport", label: "传输", value: transport },
    { key: "inference", label: "推理", value: inference },
    { key: "gap", label: "等待", value: gap },
  ].filter((item) => item.value !== null && item.value !== undefined);

  if (!parts.length) {
    return "当前步没有完整的时延数据。";
  }

  const main = parts.reduce((best, item) => (Number(item.value) > Number(best.value) ? item : best), parts[0]);
  if (main.key === "transport") {
    return "当前帧主要卡在传输链路，更像相机采集、编码、网络或 IPC 堵塞。";
  }
  if (main.key === "inference") {
    return "当前帧主要卡在推理阶段，更像模型计算或前后处理拥塞。";
  }
  return "总时延高于主要阶段之和，说明链路里还有未显式记录的等待时间。";
}

function actionColor(label, index) {
  const lower = String(label || "").toLowerCase();
  if (lower === "x") return "#0f766e";
  if (lower === "y") return "#c2410c";
  if (lower === "z") return "#2563eb";
  if (lower.includes("gripper") || lower.includes("grip") || lower.includes("hand")) return "#16a34a";
  if (lower.startsWith("q") || lower.startsWith("r")) return "#7c3aed";
  return pickColor(index);
}

function actionDimensionCards(chunk = [], labels = []) {
  if (!chunk.length) {
    return [];
  }
  const dimCount = chunk[0]?.length || 0;
  return Array.from({ length: dimCount }, (_, idx) => {
    const values = chunk.map((row) => row?.[idx] ?? null);
    const valid = finiteNumbers(values);
    const first = values[0] ?? null;
    const last = values[values.length - 1] ?? null;
    return {
      label: labels[idx] || `dim_${idx}`,
      color: actionColor(labels[idx], idx),
      values,
      first,
      last,
      delta:
        first === null || last === null || first === undefined || last === undefined
          ? null
          : Number(last) - Number(first),
      span: valid.length ? Math.max(...valid) - Math.min(...valid) : null,
    };
  });
}

function vectorMagnitude(values = [], take = 3) {
  const slice = (values || []).slice(0, take);
  if (!slice.length || slice.some((value) => value === null || value === undefined || Number.isNaN(Number(value)))) {
    return null;
  }
  return Math.sqrt(slice.reduce((sum, value) => sum + Number(value) ** 2, 0));
}

function dominantActionCard(cards = []) {
  const ranked = cards.filter((item) => item.span !== null && item.span !== undefined);
  if (!ranked.length) {
    return null;
  }
  return ranked.reduce((best, item) => (Number(item.span) > Number(best.span) ? item : best), ranked[0]);
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

function mergeStepDetails(previous = [], totalSteps = 0, incoming = []) {
  const next = Array.from({ length: Math.max(totalSteps, previous.length) }, (_, idx) => previous[idx] || null);
  for (const step of incoming || []) {
    const idx = Number(step?.step_idx);
    if (Number.isInteger(idx) && idx >= 0 && idx < next.length) {
      next[idx] = step;
    }
  }
  return next;
}

function timingFromSeries(replay, idx) {
  const series = replay?.timing_series || {};
  return {
    transport_latency_ms: series.transport_latency_ms?.[idx] ?? null,
    inference_latency_ms: series.inference_latency_ms?.[idx] ?? null,
    total_latency_ms: series.total_latency_ms?.[idx] ?? null,
    message_interval_ms: series.message_interval_ms?.[idx] ?? null,
  };
}

function fallbackStep(replay, idx) {
  return {
    step_idx: idx,
    prompt: replay?.meta?.task_prompt || null,
    tags: {},
    state: replay?.states?.[idx] || [],
    action_preview: replay?.first_actions?.[idx] || [],
    action_chunk: [],
    timing: timingFromSeries(replay, idx),
    images: [],
    raw_step: null,
  };
}

function hasStepDetail(step) {
  return Boolean(step && step.step_idx !== undefined);
}

function windowKey(center, radius) {
  return `${Number(center)}:${Number(radius)}`;
}

function parseWindowKey(key) {
  const [center, radius] = String(key).split(":").map((value) => Number(value));
  return { center, radius };
}

function windowKeyCoversStep(key, stepIdx, maxStep) {
  const { center, radius } = parseWindowKey(key);
  if (!Number.isFinite(center) || !Number.isFinite(radius)) {
    return false;
  }
  return stepIdx >= Math.max(0, center - radius) && stepIdx <= Math.min(maxStep, center + radius);
}

function rerunViewerHref(status) {
  if (typeof window === "undefined" || !status) {
    return "";
  }
  if (status.viewer_url) {
    return status.viewer_url;
  }
  if (status.server?.viewer_url) {
    return status.server.viewer_url;
  }
  const recordingUrl = status.recording_url || status.server?.recording_url;
  const viewerBase = String(
    status.web_viewer_url || status.server?.web_viewer_url || process.env.NEXT_PUBLIC_RERUN_VIEWER_BASE_URL || "https://app.rerun.io"
  ).replace(/\/$/, "");
  if (recordingUrl) {
    return `${viewerBase}/?url=${encodeURIComponent(recordingUrl)}`;
  }
  if (status.url) {
    const rrdUrl = new URL(status.url, window.location.origin).toString();
    return `${viewerBase}/?url=${encodeURIComponent(rrdUrl)}`;
  }
  return "";
}

function stepWindow(stepIdx, maxStep, radius = 4) {
  const start = Math.max(0, stepIdx - radius);
  const end = Math.min(maxStep, stepIdx + radius);
  return Array.from({ length: end - start + 1 }, (_, idx) => start + idx);
}

function executionRows(replay, boundary) {
  const actions = replay.expanded_execution?.expanded_actions || [];
  const start = boundary?.start || 0;
  const end = boundary?.end || start;
  return actions.slice(start, end).map((row, index) => ({
    execStep: start + index,
    values: row,
  }));
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

function MiniTrend({ values = [], color = "#0f766e", activeIndex = null }) {
  const width = 180;
  const height = 54;
  const padding = 6;
  const valid = values
    .map((value, index) => ({
      value: value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value),
      index,
    }))
    .filter((item) => item.value !== null);

  if (!valid.length) {
    return <div className="mini-trend-empty">无趋势</div>;
  }

  const numbers = valid.map((item) => item.value);
  const min = Math.min(...numbers);
  const max = Math.max(...numbers);
  const span = max - min || 1;
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;
  const points = valid.map((item) => {
    const x = padding + (usableWidth * item.index) / Math.max(values.length - 1, 1);
    const y = height - padding - ((item.value - min) / span) * usableHeight;
    return { x, y };
  });
  const pointText = points.map((point) => `${point.x},${point.y}`).join(" ");
  const activePoint =
    activeIndex === null
      ? null
      : valid.find((item) => item.index === activeIndex);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="mini-trend">
      <rect x="0" y="0" width={width} height={height} rx="14" fill="rgba(255,255,255,0.72)" />
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="3"
        strokeLinejoin="round"
        strokeLinecap="round"
        points={pointText}
      />
      {activePoint ? (
        <circle
          cx={padding + (usableWidth * activePoint.index) / Math.max(values.length - 1, 1)}
          cy={height - padding - ((activePoint.value - min) / span) * usableHeight}
          r="4"
          fill={color}
          stroke="white"
          strokeWidth="2"
        />
      ) : null}
    </svg>
  );
}

function LatencyBreakdown({ timing }) {
  const { transport, inference, gap, total } = timingSnapshot(timing);
  const segments = [
    { key: "transport", label: "传输", value: transport, className: "is-transport" },
    { key: "inference", label: "推理", value: inference, className: "is-inference" },
    { key: "gap", label: "等待", value: gap, className: "is-gap" },
  ].filter((item) => item.value !== null && item.value !== undefined && Number(item.value) > 0);

  if (!segments.length) {
    return <div className="empty-panel">没有可分解的时延数据</div>;
  }

  return (
    <div className="breakdown-shell">
      <div className="breakdown-bar">
        {segments.map((segment) => (
          <span
            key={segment.key}
            className={`breakdown-segment ${segment.className}`}
            style={{ flexGrow: Math.max(Number(segment.value), 1) }}
            title={`${segment.label}: ${formatMs(segment.value)}`}
          />
        ))}
      </div>
      <div className="breakdown-legend">
        {segments.map((segment) => (
          <span key={segment.key} className="breakdown-legend-item">
            <span className={`breakdown-dot ${segment.className}`} />
            {segment.label} {formatMs(segment.value)} · {percentText(ratio(segment.value, total))}
          </span>
        ))}
      </div>
    </div>
  );
}

function NearbyLatencyBars({ rows = [], currentStepIdx, onSelect }) {
  const maxValue = Math.max(1, ...rows.map((row) => row.total ?? 0));

  if (!rows.length) {
    return <div className="empty-panel">没有附近时延窗口</div>;
  }

  return (
    <div className="nearby-latency-grid">
      {rows.map((row) => (
        <button
          key={`latency-${row.stepIdx}`}
          type="button"
          className={`nearby-latency-card${row.stepIdx === currentStepIdx ? " is-active" : ""}`}
          onClick={() => onSelect(row.stepIdx)}
        >
          <div
            className="nearby-latency-bar"
            style={{ height: `${24 + ((row.total ?? 0) / maxValue) * 60}px` }}
          />
          <strong>步 {row.stepIdx}</strong>
          <span>{formatMs(row.total)}</span>
        </button>
      ))}
    </div>
  );
}

function FrameDiagnosisPanel({
  currentStep,
  stepIdx,
  nearbyLatencyRows,
  latencySummary,
  meta,
  onSelectStep,
}) {
  const timing = timingSnapshot(currentStep.timing);
  const totalStatus = latencyTone(timing.total, latencySummary?.total_latency);
  const totalValues = nearbyLatencyRows.map((item) => item.total);
  const totalAvg = latencySummary?.total_latency?.avg_ms ?? meanValue(totalValues);
  const totalRank = percentileRank(totalValues, timing.total);
  const promptText = currentStep.prompt || meta.task_prompt || "--";
  const tags = Object.entries(currentStep.tags || {});
  const cards = [
    {
      label: "总延迟",
      value: formatMs(timing.total),
      tone: totalStatus,
      note:
        totalAvg === null || totalAvg === undefined
          ? "缺少全局均值"
          : `较均值 ${signedNumber(timing.total - totalAvg)} ms`,
    },
    {
      label: "传输占比",
      value: percentText(ratio(timing.transport, timing.total)),
      tone: latencyTone(timing.transport, latencySummary?.transport_latency),
      note: formatMs(timing.transport),
    },
    {
      label: "推理占比",
      value: percentText(ratio(timing.inference, timing.total)),
      tone: latencyTone(timing.inference, latencySummary?.inference_latency),
      note: formatMs(timing.inference),
    },
    {
      label: "全局分位",
      value: totalRank === null ? "--" : `P${Math.round(totalRank)}`,
      tone: totalStatus,
      note: totalRank === null ? "无法比较" : "数值越高说明越慢",
    },
  ];

  return (
    <div className="section-panel frame-diagnosis-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">当前帧</p>
          <h2>延迟、提示词与时间线</h2>
        </div>
        <span className="frame-badge">步 {stepIdx}</span>
      </div>

      <div className="diagnostic-card-grid">
        {cards.map((card) => (
          <article key={card.label} className="diagnostic-card">
            <div className="diagnostic-card-top">
              <span className="stat-label">{card.label}</span>
              <span className={`diagnostic-chip ${card.tone.toneClass}`}>{card.tone.label}</span>
            </div>
            <strong>{card.value}</strong>
            <p>{card.note}</p>
          </article>
        ))}
      </div>

      <div className="frame-diagnosis-grid">
        <article className="context-card">
          <p className="panel-title">时延拆解</p>
          <LatencyBreakdown timing={currentStep.timing} />
          <p className="context-note">{latencyDriver(currentStep.timing)}</p>
        </article>

        <article className="context-card">
          <p className="panel-title">提示词与标签</p>
          <p className="prompt-block">{promptText}</p>
          {tags.length ? (
            <div className="tag-list">
              {tags.map(([key, value]) => (
                <span key={key} className="meta-tag">{key}: {String(value)}</span>
              ))}
            </div>
          ) : (
            <p className="context-note">当前 step 没有额外标签。</p>
          )}
        </article>
      </div>

      <article className="context-card">
        <div className="context-card-header">
          <p className="panel-title">附近延迟窗口</p>
          <span className="muted">点击切换到对应帧</span>
        </div>
        <NearbyLatencyBars rows={nearbyLatencyRows} currentStepIdx={stepIdx} onSelect={onSelectStep} />
      </article>

      {currentStep.timeline_events?.length ? (
        <TimelineEvents events={currentStep.timeline_events} />
      ) : (
        <div className="timeline-fallback">
          <strong>未记录原始链路时间戳</strong>
          <p>这次 run 只有 transport / inference 时长，没有 client_send、server_recv、infer_start 等事件点。</p>
        </div>
      )}
    </div>
  );
}

function ActionWindowPanel({ currentStep, currentBoundary, currentExecution, actionLabels = [] }) {
  const chunk = currentStep.action_chunk || [];
  const cards = actionDimensionCards(chunk, actionLabels);
  const dominant = dominantActionCard(cards);
  const firstXYZ = chunk[0]?.slice(0, 3) || [];
  const lastXYZ = chunk[chunk.length - 1]?.slice(0, 3) || [];
  const xyzShift =
    firstXYZ.length === 3 && lastXYZ.length === 3
      ? Math.sqrt(firstXYZ.reduce((sum, value, index) => sum + (Number(lastXYZ[index]) - Number(value)) ** 2, 0))
      : null;
  const gripperIdx = actionLabels.findIndex((label) => String(label).toLowerCase() === "gripper");
  const gripperEnd = gripperIdx >= 0 ? chunk[chunk.length - 1]?.[gripperIdx] ?? null : null;

  const summaryCards = [
    { label: "窗口长度", value: chunk.length ? `${chunk.length} 步` : "--", note: `维度 ${actionLabels.length || 0}` },
    {
      label: "执行区间",
      value: currentExecution.length ? `${currentBoundary.start}-${currentBoundary.end - 1}` : "--",
      note: currentExecution.length ? "全局执行步" : "暂无展开执行数据",
    },
    {
      label: "XYZ 漂移",
      value: xyzShift === null ? "--" : formatNumber(xyzShift, 3),
      note: xyzShift === null ? "缺少三维动作" : "首步到末步的欧氏距离",
    },
    {
      label: "主变化维度",
      value: dominant?.label || "--",
      note: dominant?.span === null || dominant?.span === undefined ? "无法判断" : `波动 ${formatNumber(dominant.span, 3)}`,
    },
  ];

  return (
    <div className="section-panel action-window-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">动作块</p>
          <h2>当前推理窗口</h2>
        </div>
        {chunk.length ? (
          <span className="frame-badge">
            {chunk.length} 步 × {(chunk[0] || []).length} 维
          </span>
        ) : null}
      </div>

      <div className="diagnostic-card-grid">
        {summaryCards.map((card) => (
          <article key={card.label} className="diagnostic-card">
            <span className="stat-label">{card.label}</span>
            <strong>{card.value}</strong>
            <p>{card.note}</p>
          </article>
        ))}
      </div>

      <article className="context-card">
        <div className="context-card-header">
          <p className="panel-title">维度趋势</p>
          <span className="muted">每张卡表示这次 chunk 内一个动作维度的变化</span>
        </div>
        {cards.length ? (
          <div className="action-dimension-grid">
            {cards.map((card) => (
              <article key={card.label} className="action-dimension-card">
                <div className="action-dimension-head">
                  <strong>{card.label}</strong>
                  <span className={card.delta !== null && card.delta < 0 ? "tone-bad" : "tone-good"}>
                    {signedNumber(card.delta, 3)}
                  </span>
                </div>
                <MiniTrend values={card.values} color={card.color} activeIndex={0} />
                <div className="action-dimension-meta">
                  <span>首 {formatNumber(card.first, 3)}</span>
                  <span>末 {formatNumber(card.last, 3)}</span>
                  <span>波动 {formatNumber(card.span, 3)}</span>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-panel">当前 step 没有动作块。</div>
        )}
      </article>
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
  const [activeTab, setActiveTab] = useState("replay");
  const [globalView, setGlobalView] = useState("inference");
  const [stepIdx, setStepIdx] = useState(0);
  const [stepInput, setStepInput] = useState("0");
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackMs, setPlaybackMs] = useState(220);
  const [selectedCamera, setSelectedCamera] = useState("all");
  const [railCollapsed, setRailCollapsed] = useState(false);
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
  const [stepDetails, setStepDetails] = useState(() => replay.step_details || []);
  const [pendingWindowKeys, setPendingWindowKeys] = useState([]);
  const [stepWindowError, setStepWindowError] = useState("");
  const [rerunStatus, setRerunStatus] = useState(replay.rerun_status || null);
  const [rerunError, setRerunError] = useState("");
  const [isBuildingRerun, setIsBuildingRerun] = useState(false);
  const [, startTransition] = useTransition();
  const lastPlaybackTickRef = useRef(null);
  const pendingWindowKeysRef = useRef(new Set());
  const requestedWindowKeysRef = useRef(new Set());

  const summary = replay.summary;
  const meta = replay.meta || {};
  const totalSteps = summary.total_steps ?? stepDetails.length;
  const maxStep = Math.max(0, totalSteps - 1);
  const steps = stepDetails;
  const currentStep = steps[stepIdx] || fallbackStep(replay, stepIdx);
  const currentBoundary = currentStep.action_chunk?.length
    ? { start: stepIdx, end: stepIdx + currentStep.action_chunk.length }
    : currentChunkBoundary(replay.expanded_execution, stepIdx);
  const nearbySteps = stepWindow(stepIdx, maxStep, 5);
  const allCameras = useMemo(() => replay.camera_names || [], [replay.camera_names]);
  const currentExecution = useMemo(() => executionRows(replay, currentBoundary), [replay, currentBoundary]);
  const playbackOptions = [
    { label: "慢", value: 500 },
    { label: "中", value: 220 },
    { label: "快", value: 120 },
  ];

  useEffect(() => {
    setStepDetails(replay.step_details || []);
    setRerunStatus(replay.rerun_status || null);
    pendingWindowKeysRef.current = new Set();
    requestedWindowKeysRef.current = new Set();
    setPendingWindowKeys([]);
    setStepWindowError("");
    setStepIdx(0);
    setStepInput("0");
  }, [replay]);

  const requestStepWindow = useCallback(
    async (center, radius = 16) => {
      const safeCenter = clamp(Number(center), 0, maxStep);
      const safeRadius = Math.max(0, Number(radius) || 0);
      const key = windowKey(safeCenter, safeRadius);
      if (pendingWindowKeysRef.current.has(key) || requestedWindowKeysRef.current.has(key)) {
        return;
      }

      pendingWindowKeysRef.current.add(key);
      requestedWindowKeysRef.current.add(key);
      setPendingWindowKeys(Array.from(pendingWindowKeysRef.current));
      setStepWindowError("");

      try {
        const data = await browserFetchJson(
          `/api/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/replay/window`,
          { center: safeCenter, radius: safeRadius }
        );
        setStepDetails((previous) => mergeStepDetails(previous, totalSteps, data?.step_details || []));
      } catch (error) {
        requestedWindowKeysRef.current.delete(key);
        setStepWindowError(String(error.message || error));
      } finally {
        pendingWindowKeysRef.current.delete(key);
        setPendingWindowKeys(Array.from(pendingWindowKeysRef.current));
      }
    },
    [maxStep, project, runName, totalSteps]
  );

  const isStepFrameReady = useCallback(
    (idx) => {
      const detail = stepDetails[idx];
      return hasStepDetail(detail);
    },
    [stepDetails]
  );

  useEffect(() => {
    if (!isStepFrameReady(stepIdx)) {
      requestStepWindow(stepIdx, 16);
    }
  }, [isStepFrameReady, requestStepWindow, stepIdx]);

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
      lastPlaybackTickRef.current = null;
      return undefined;
    }

    let frameId = 0;
    function tick(now) {
      if (lastPlaybackTickRef.current === null) {
        lastPlaybackTickRef.current = now;
      }

      const elapsed = now - lastPlaybackTickRef.current;
      if (elapsed >= playbackMs) {
        const framesToAdvance = Math.max(1, Math.floor(elapsed / playbackMs));
        lastPlaybackTickRef.current = now - (elapsed % playbackMs);
        setStepIdx((value) => {
          const nextStep = (value + framesToAdvance) % (maxStep + 1);
          if (!isStepFrameReady(nextStep)) {
            requestStepWindow(nextStep, 16);
            return value;
          }
          return nextStep;
        });
      }

      frameId = window.requestAnimationFrame(tick);
    }

    frameId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frameId);
  }, [isPlaying, isStepFrameReady, maxStep, playbackMs, requestStepWindow]);

  useEffect(() => {
    if (!isPlaying || maxStep <= 0) {
      return undefined;
    }

    for (const offset of [12, 28, 44]) {
      const targetStep = clamp(stepIdx + offset, 0, maxStep);
      if (!isStepFrameReady(targetStep)) {
        requestStepWindow(targetStep, 18);
      }
    }
    return undefined;
  }, [isPlaying, isStepFrameReady, maxStep, requestStepWindow, stepIdx]);

  useEffect(() => {
    if (!isPlaying || typeof window === "undefined") {
      return;
    }

    for (const idx of stepWindow(clamp(stepIdx + 6, 0, maxStep), maxStep, 8)) {
      for (const image of stepDetails[idx]?.images || []) {
        const src = toPublicApiUrl(image.url || image.overlay_url || "");
        if (src) {
          const preload = new window.Image();
          preload.src = src;
        }
      }
    }
  }, [isPlaying, maxStep, stepDetails, stepIdx]);

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

  async function handleBuildRerun() {
    setRerunError("");
    setIsBuildingRerun(true);
    try {
      const status = await browserPostJson(
        `/api/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/rerun/build`,
        {}
      );
      setRerunStatus(status);
    } catch (error) {
      setRerunError(String(error.message || error));
    } finally {
      setIsBuildingRerun(false);
    }
  }

  async function handleOpenRerun() {
    setRerunError("");
    setIsBuildingRerun(true);
    const popup = window.open("about:blank", "_blank");
    try {
      const status = await browserPostJson(
        `/api/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/rerun/open`,
        {}
      );
      setRerunStatus(status);
      if (status.server?.ready === false) {
        throw new Error("Rerun viewer is still starting. Please try Open in Rerun again after the proxy is ready.");
      }
      const href = rerunViewerHref(status);
      if (href) {
        if (popup) {
          popup.location.href = href;
        } else {
          window.location.href = href;
        }
      } else if (popup) {
        popup.close();
      }
    } catch (error) {
      if (popup) {
        popup.close();
      }
      setRerunError(String(error.message || error));
    } finally {
      setIsBuildingRerun(false);
    }
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

  const workspaceMode =
    activeTab === "latency" || activeTab === "action"
      ? "analyze"
      : activeTab;
  const pendingCurrentFrame = pendingWindowKeys.some((key) => windowKeyCoversStep(key, stepIdx, maxStep));
  const currentHasDetail = hasStepDetail(steps[stepIdx]);
  const currentHasImages = Boolean(currentStep.images?.length);
  const frameStatus = !currentHasDetail
    ? { state: "loading", label: `Loading exact frame ${stepIdx}`, detail: "Fetching the replay window for this step." }
    : allCameras.length && !currentHasImages && pendingCurrentFrame
      ? { state: "loading", label: `Loading exact frame ${stepIdx}`, detail: "Waiting for camera artifacts in the requested window." }
      : allCameras.length && !currentHasImages
        ? { state: "missing", label: `No images recorded for step ${stepIdx}`, detail: "The step detail is loaded, but it contains no camera images." }
        : { state: "ready", label: `Frame ${stepIdx} ready`, detail: `${currentStep.images?.length || 0} camera image(s) loaded.` };
  const runPrompt = meta.task_prompt || summary.task_name || currentStep.prompt || "--";
  const states = replay.states || [];
  const firstActions = replay.first_actions || [];
  const timingSeries = replay.timing_series || {};
  const trajectory = {
    title: "历史轨迹 + 预测",
    currentPoint: currentStep.state,
    series: [
      {
        name: "历史",
        color: "#4b5563",
        points: states.slice(0, stepIdx + 1),
      },
      {
        name: "预测",
        color: "#dc2626",
        dashed: true,
        points: currentStep.action_chunk,
      },
    ],
  };
  const latencyRows = nearbySteps.map((idx) => ({
    stepIdx: idx,
    total: totalLatency((steps[idx] || fallbackStep(replay, idx)).timing || {}),
  }));
  const replaySignalPanels = useMemo(() => {
    const stateXyz = groupByKey(replay.state_groups || [], "xyz", [0, 1, 2], "XYZ");
    const actionXyz = groupByKey(replay.action_groups || [], "xyz", [0, 1, 2], "XYZ");
    const stateGripperIdx = labelIndex(replay.state_labels || [], ["gripper", "grip"]);
    const actionGripperIdx = labelIndex(replay.action_labels || [], ["gripper", "grip"]);
    const gripperSeries = [
      stateGripperIdx >= 0
        ? {
          name: "state",
          values: states.map((row) => row?.[stateGripperIdx] ?? null),
          color: "#0f766e",
        }
        : null,
      actionGripperIdx >= 0
        ? {
          name: "action",
          values: firstActions.map((row) => row?.[actionGripperIdx] ?? null),
          color: "#c2410c",
        }
        : null,
    ].filter(Boolean);

    return [
      {
        title: "State XYZ",
        markerIndex: stepIdx,
        series: seriesForGroup(states, replay.state_labels || [], stateXyz).slice(0, 3),
      },
      {
        title: "Action XYZ",
        markerIndex: stepIdx,
        series: seriesForGroup(firstActions, replay.action_labels || [], actionXyz).slice(0, 3),
      },
      {
        title: "Gripper",
        markerIndex: stepIdx,
        series: gripperSeries,
      },
      {
        title: "Latency",
        markerIndex: stepIdx,
        unit: "ms",
        series: [
          { name: "total", values: timingSeries.total_latency_ms || [], color: "#6b7280" },
          { name: "inference", values: timingSeries.inference_latency_ms || [], color: "#2563eb" },
        ],
      },
    ].filter((panel) => panel.series?.some((item) => item.values?.some((value) => value !== null && value !== undefined)));
  }, [
    firstActions,
    replay.action_groups,
    replay.action_labels,
    replay.state_groups,
    replay.state_labels,
    states,
    stepIdx,
    timingSeries.inference_latency_ms,
    timingSeries.total_latency_ms,
  ]);
  const attentionControls = {
    device: attentionDevice,
    layer: attentionLayer,
    modelPath: attentionModelPath,
    prompt: attentionPrompt,
    stdoutTail: attentionStdoutTail,
    isGenerating: isGeneratingAttention,
    onDeviceChange: setAttentionDevice,
    onLayerChange: setAttentionLayer,
    onModelPathChange: setAttentionModelPath,
    onPromptChange: setAttentionPrompt,
    onGenerateCurrent: () => handleGenerateAttention("current"),
  };
  const currentLatency = totalLatency(currentStep.timing || {});
  const averageLatency = meanValue(timingSeries.total_latency_ms || []);
  const p95Latency = replay.latency_summary?.total_latency?.p95_ms ?? null;
  const currentLatencyRank = percentileRank(timingSeries.total_latency_ms || [], currentLatency);
  const analysisKpis = [
    { label: "Current total", value: formatMs(currentLatency), note: `${percentText(currentLatencyRank)} percentile` },
    { label: "Run average", value: formatMs(averageLatency), note: `${signedNumber(Number(currentLatency || 0) - Number(averageLatency || 0), 1)} ms vs avg` },
    { label: "P95 total", value: formatMs(p95Latency), note: "run latency tail" },
    { label: "Action dims", value: summary.action_dim ?? replay.action_labels?.length ?? "--", note: `${summary.total_steps ?? 0} steps` },
  ];
  const analysisCharts =
    globalView === "inference"
      ? [
        ...(replay.state_groups || []).map((group) => ({
          key: `state-${group.key}`,
          title: `State / ${group.title}`,
          xLabel: "inference step",
          series: group.indices?.length ? seriesForGroup(states, replay.state_labels, group) : [],
        })),
        ...(replay.action_groups || []).map((group) => ({
          key: `action-${group.key}`,
          title: `Action / ${group.title}`,
          xLabel: "inference step",
          series: group.indices?.length ? seriesForGroup(firstActions, replay.action_labels, group) : [],
        })),
      ]
      : (replay.expanded_execution?.action_groups || []).map((group) => ({
        key: `exec-${group.key}`,
        title: `Executed action / ${group.title}`,
        xLabel: "execution step",
        series: group.indices?.length
          ? seriesForGroup(
            replay.expanded_execution.expanded_actions,
            replay.expanded_execution.action_labels,
            group
          )
          : [],
      }));
  const analyzeContent = (
    <div className="run-workspace-analysis">
      <section className="run-workspace-analysis-hero">
        <div>
          <p className="run-workspace-eyebrow">Analyze</p>
          <h2>Latency and action dashboard</h2>
          <p>{runPrompt}</p>
        </div>
        <div className="run-workspace-analysis-kpis">
          {analysisKpis.map((item) => (
            <article key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
              <p>{item.note}</p>
            </article>
          ))}
        </div>
      </section>

      <div className="run-workspace-analysis-tabs">
        <button
          type="button"
          className={`run-workspace-mode-button${globalView === "inference" ? " run-workspace-mode-button-active" : ""}`}
          onClick={() => setGlobalView("inference")}
        >
          Inference
        </button>
        <button
          type="button"
          className={`run-workspace-mode-button${globalView === "execution" ? " run-workspace-mode-button-active" : ""}`}
          onClick={() => setGlobalView("execution")}
        >
          Execution
        </button>
      </div>

      <div className="run-workspace-analysis-grid run-workspace-analysis-grid-top">
        <LineChart
          title="Latency timeline"
          markerIndex={stepIdx}
          xLabel="step"
          series={[
            { name: "transport", values: timingSeries.transport_latency_ms || [], color: "#d97706" },
            { name: "inference", values: timingSeries.inference_latency_ms || [], color: "#2563eb" },
            { name: "total", values: timingSeries.total_latency_ms || [], color: "#6b7280" },
            { name: "message", values: timingSeries.message_interval_ms || [], color: "#0f766e" },
          ]}
        />
        <HistogramChart title="Total latency distribution" values={timingSeries.total_latency_ms || []} color="#475569" />
      </div>

      <div className="run-workspace-chart-grid">
        {analysisCharts.map((chart) =>
          chart.series?.length ? (
            <LineChart
              key={chart.key}
              title={chart.title}
              markerIndex={globalView === "execution" ? currentBoundary.start : stepIdx}
              xLabel={chart.xLabel}
              series={chart.series}
            />
          ) : null
        )}
      </div>
    </div>
  );

  return (
    <div className="run-workspace-shell">
      <header className="run-workspace-header">
        <div className="run-workspace-header-main">
          <p className="run-workspace-eyebrow">{summary.project}</p>
          <h1>{summary.run_name}</h1>
          <p>{runPrompt}</p>
        </div>
        <div className="run-workspace-header-metrics">
          <span><strong>{summary.model_name || "unknown"}</strong> model</span>
          <span><strong>{summary.total_steps ?? 0}</strong> steps</span>
          {summary.inference_freq ? <span><strong>{summary.inference_freq}</strong> Hz</span> : null}
          {summary.action_dim ? <span><strong>{summary.action_dim}</strong> actions</span> : null}
          <span><strong>{formatDate(summary.updated_at)}</strong> updated</span>
        </div>
        <div className="run-workspace-header-actions">
          <button type="button" className="run-workspace-button" disabled={isBuildingRerun} onClick={handleOpenRerun}>
            {isBuildingRerun ? "Opening" : rerunStatus?.available && !rerunStatus?.stale ? "Open in Rerun" : "Build + Open Rerun"}
          </button>
          <button type="button" className="run-workspace-button run-workspace-button-secondary" disabled={isBuildingRerun} onClick={handleBuildRerun}>
            Rebuild .rrd
          </button>
        </div>
        {rerunError || stepWindowError || attentionError ? (
          <p className="run-workspace-error">{rerunError || stepWindowError || attentionError}</p>
        ) : null}
      </header>

      <div className="run-workspace-grid">
        <RunWorkspaceRail
          mode={workspaceMode}
          onModeChange={setActiveTab}
          collapsed={railCollapsed}
          onToggleCollapsed={() => setRailCollapsed((value) => !value)}
          summary={summary}
          currentStep={currentStep}
          stepIdx={stepIdx}
          maxStep={maxStep}
          frameStatus={frameStatus}
          allCameras={allCameras}
          selectedCamera={selectedCamera}
          onCameraChange={setSelectedCamera}
          rerunStatus={rerunStatus}
        />

        <RunWorkspaceViewer
          mode={workspaceMode}
          currentStep={currentStep}
          selectedCamera={selectedCamera}
          onCameraChange={setSelectedCamera}
          allCameras={allCameras}
          stepIdx={stepIdx}
          maxStep={maxStep}
          isPlaying={isPlaying}
          playbackMs={playbackMs}
          playbackOptions={playbackOptions}
          stepInput={stepInput}
          onStepChange={jumpToStep}
          onPlayToggle={() => setIsPlaying((value) => !value)}
          onPlaybackMsChange={setPlaybackMs}
          onStepInputChange={setStepInput}
          onStepInputCommit={jumpToStep}
          frameStatus={frameStatus}
          attentionData={attentionData}
          attentionLoading={isLoadingAttention || isGeneratingAttention}
          attentionControls={attentionControls}
          trajectory={trajectory}
          signalPanels={replaySignalPanels}
          showTrajectory={false}
        >
          {analyzeContent}
        </RunWorkspaceViewer>

        <RunWorkspaceInspector
          mode={workspaceMode}
          summary={summary}
          meta={meta}
          replay={replay}
          currentStep={currentStep}
          stepIdx={stepIdx}
          currentBoundary={currentBoundary}
          currentExecution={currentExecution}
          latencySummary={replay.latency_summary || {}}
        />
      </div>
    </div>
  );
}
