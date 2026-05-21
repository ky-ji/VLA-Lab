"use client";

import { useEffect, useState } from "react";

import RunReplayClient from "@/components/run-replay-client";
import { browserFetchJson } from "@/lib/api";

const SERIES_ENTITIES = [
  "/robot/state",
  "/policy/action_first",
  "/latency/transport_ms",
  "/latency/inference_ms",
  "/latency/total_ms",
];

function emptyArray(length) {
  return Array.from({ length: Math.max(0, Number(length) || 0) }, () => null);
}

function sparseSteps(totalSteps, windowPayload) {
  const steps = emptyArray(totalSteps);
  for (const step of windowPayload?.step_details || []) {
    const idx = Number(step.step_idx);
    if (Number.isInteger(idx) && idx >= 0 && idx < steps.length) {
      steps[idx] = step;
    }
  }
  return steps;
}

function normalizeSeries(seriesPayload, totalSteps) {
  const entities = seriesPayload?.entities || {};
  const nulls = emptyArray(totalSteps);
  return {
    states: entities["/robot/state"]?.values || nulls,
    first_actions: entities["/policy/action_first"]?.values || nulls,
    timing_series: {
      transport_latency_ms: entities["/latency/transport_ms"]?.values || nulls,
      inference_latency_ms: entities["/latency/inference_ms"]?.values || nulls,
      total_latency_ms: entities["/latency/total_ms"]?.values || nulls,
      message_interval_ms: nulls,
    },
  };
}

function replayFromProgressivePayload(summaryPayload, windowPayload, seriesPayload, rerunStatus) {
  const totalSteps = summaryPayload?.summary?.total_steps || windowPayload?.total_steps || 0;
  const series = normalizeSeries(seriesPayload, totalSteps);
  const chunkBoundaries = emptyArray(totalSteps).map((_, idx) => ({ start: idx, end: idx }));
  return {
    summary: summaryPayload?.summary || {},
    meta: summaryPayload?.meta || {},
    camera_names: summaryPayload?.camera_names || windowPayload?.camera_names || [],
    state_labels: summaryPayload?.state_labels || [],
    action_labels: summaryPayload?.action_labels || [],
    state_groups: summaryPayload?.state_groups || [],
    action_groups: summaryPayload?.action_groups || [],
    states: series.states,
    first_actions: series.first_actions,
    timing_series: series.timing_series,
    expanded_execution: {
      expanded_actions: [],
      chunk_boundaries: chunkBoundaries,
      action_labels: summaryPayload?.action_labels || [],
      action_groups: summaryPayload?.action_groups || [],
    },
    latency_summary: {},
    action_stats: [],
    step_details: sparseSteps(totalSteps, windowPayload),
    attention_caches: [],
    index: summaryPayload?.index || windowPayload?.index || null,
    rerun_status: rerunStatus || null,
  };
}

export default function RunDetailPageClient({ project, runName }) {
  const [replay, setReplay] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const base = `/api/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}`;

    async function loadReplay() {
      setLoading(true);
      setError("");

      try {
        const summaryPayload = await browserFetchJson(`${base}/replay/summary`);
        const totalSteps = summaryPayload?.summary?.total_steps || 0;
        const [windowPayload, seriesPayload, rerunStatus] = await Promise.all([
          browserFetchJson(`${base}/replay/window`, { center: 0, radius: 16 }),
          browserFetchJson(`${base}/series`, {
            entities: SERIES_ENTITIES,
            start: 0,
            end: totalSteps,
            stride: 1,
          }),
          browserFetchJson(`${base}/rerun/status`),
        ]);
        if (!cancelled) {
          setReplay(replayFromProgressivePayload(summaryPayload, windowPayload, seriesPayload, rerunStatus));
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : "加载 run 详情失败");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadReplay();
    return () => {
      cancelled = true;
    };
  }, [project, runName]);

  if (loading) {
    return (
      <section className="placeholder-panel">
        <p>正在加载 run 索引与首个窗口 ...</p>
      </section>
    );
  }

  if (error || !replay) {
    return (
      <section className="empty-panel">
        <p>{error || "未找到该 run 的详情数据"}</p>
      </section>
    );
  }

  return <RunReplayClient replay={replay} project={project} runName={runName} />;
}
