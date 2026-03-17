"use client";

import { useEffect, useMemo, useState } from "react";

import { browserFetchJson } from "@/lib/api";
import { formatMs } from "@/lib/format";
import { HistogramChart, LineChart, SimpleTabs } from "@/components/chart-kit";
import RunTable from "@/components/run-table";

const PALETTE = ["#0f766e", "#c2410c", "#2563eb", "#7c3aed", "#0891b2", "#be123c", "#65a30d"];

function pickColor(index) {
  return PALETTE[index % PALETTE.length];
}

function metricLabel(key) {
  if (key === "transport_latency") {
    return "Transport";
  }
  if (key === "inference_latency") {
    return "Inference";
  }
  if (key === "total_latency") {
    return "Total";
  }
  return key;
}

function scoreLatency(summary) {
  const avg = summary?.total_latency?.avg_ms;
  const p95 = summary?.total_latency?.p95_ms;
  if (avg === null || avg === undefined || p95 === null || p95 === undefined) {
    return { label: "No data", tone: "muted" };
  }
  if (avg < 50 && p95 < 100) {
    return { label: "实时裕量充足", tone: "good" };
  }
  if (avg < 100 && p95 < 200) {
    return { label: "可用但需观察", tone: "warn" };
  }
  return { label: "需要优化", tone: "bad" };
}

export default function LatencyCompareClient({ initialRuns = [], initialCompare = null }) {
  const [allRuns, setAllRuns] = useState(initialRuns);
  const [compare, setCompare] = useState(initialCompare);
  const [projectFilter, setProjectFilter] = useState("");
  const [query, setQuery] = useState("");
  const [selectedRunIds, setSelectedRunIds] = useState(
    initialCompare?.items?.length
      ? initialCompare.items.map((item) => item.run.run_id)
      : initialRuns.slice(0, 3).map((run) => run.run_id)
  );
  const [activeTab, setActiveTab] = useState("series");
  const [loadingRuns, setLoadingRuns] = useState(initialRuns.length === 0);
  const [loadingCompare, setLoadingCompare] = useState(initialCompare == null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (initialRuns.length) {
      return undefined;
    }
    let ignore = false;
    setLoadingRuns(true);

    async function loadRuns() {
      try {
        const data = await browserFetchJson("/api/runs", {
          limit: 240,
        });
        if (ignore) {
          return;
        }
        const items = data?.items || [];
        setAllRuns(items);
        setSelectedRunIds((current) => (current.length ? current : items.slice(0, 3).map((run) => run.run_id)));
      } catch (err) {
        if (!ignore) {
          setError(String(err.message || err));
        }
      } finally {
        if (!ignore) {
          setLoadingRuns(false);
        }
      }
    }

    loadRuns();
    return () => {
      ignore = true;
    };
  }, [initialRuns]);

  useEffect(() => {
    if (!selectedRunIds.length) {
      setCompare({ items: [] });
      return;
    }

    let ignore = false;
    setLoadingCompare(true);
    setError("");

    async function loadCompare() {
      try {
        const data = await browserFetchJson("/api/latency/compare", {
          runs: selectedRunIds,
          max_points: 600,
        });
        if (!ignore) {
          setCompare(data);
        }
      } catch (err) {
        if (!ignore) {
          setError(String(err.message || err));
        }
      } finally {
        if (!ignore) {
          setLoadingCompare(false);
        }
      }
    }

    loadCompare();
    return () => {
      ignore = true;
    };
  }, [selectedRunIds]);

  const projects = useMemo(
    () => Array.from(new Set(allRuns.map((run) => run.project))).sort(),
    [allRuns]
  );

  const filteredRuns = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return allRuns.filter((run) => {
      if (projectFilter && run.project !== projectFilter) {
        return false;
      }
      if (!needle) {
        return true;
      }
      return [
        run.run_name,
        run.project,
        run.model_name,
        run.task_name,
        run.robot_name,
      ]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(needle));
    });
  }, [allRuns, projectFilter, query]);

  const selectedSet = new Set(selectedRunIds);
  const compareItems = compare?.items || [];
  const showComparePlaceholder = loadingCompare && compareItems.length === 0;

  function toggleRun(runId) {
    setSelectedRunIds((current) =>
      current.includes(runId) ? current.filter((value) => value !== runId) : [...current, runId]
    );
  }

  function setFirstThreeFiltered() {
    setSelectedRunIds(filteredRuns.slice(0, 3).map((run) => run.run_id));
  }

  return (
    <div className="section-stack">
      <section className="selection-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Latency Compare</p>
            <h2>多 run 时延深度分析</h2>
          </div>
        </div>
        <div className="form-grid form-grid-wide">
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索 run / project / model / task"
          />
          <select value={projectFilter} onChange={(event) => setProjectFilter(event.target.value)}>
            <option value="">全部项目</option>
            {projects.map((project) => (
              <option key={project} value={project}>
                {project}
              </option>
            ))}
          </select>
          <button type="button" onClick={setFirstThreeFiltered}>
            选择前 3 个
          </button>
        </div>
        <div className="selection-actions">
          <span className="muted">
            {loadingRuns ? "加载 run 列表中..." : `已筛出 ${filteredRuns.length} 个 run，当前选中 ${selectedRunIds.length} 个`}
          </span>
          <button type="button" className="button-ghost secondary-button" onClick={() => setSelectedRunIds([])}>
            清空选择
          </button>
        </div>
        {error ? <div className="placeholder-note">{error}</div> : null}
        <div className="checkbox-grid">
          {filteredRuns.map((run) => (
            <label key={run.run_id} className="checkbox-item">
              <input
                type="checkbox"
                checked={selectedSet.has(run.run_id)}
                onChange={() => toggleRun(run.run_id)}
              />
              <span>
                <strong>{run.run_name}</strong>
                <br />
                <span className="muted">{run.project}</span>
              </span>
            </label>
          ))}
        </div>
      </section>

      <section className="selection-panel">
        <SimpleTabs
          activeTab={activeTab}
          onChange={setActiveTab}
          tabs={[
            { key: "series", label: "时序图" },
            { key: "distribution", label: "统计分布" },
            { key: "detail", label: "详细对比" },
          ]}
        />
      </section>

      {activeTab === "series" ? (
        <section className="section-stack">
          {showComparePlaceholder ? (
            <div className="placeholder-panel">
              <p>正在加载时延对比 ...</p>
            </div>
          ) : (
            <>
              <LineChart
                title="Transport latency"
                xLabel="X 轴: step"
                series={compareItems.map((item, index) => ({
                  name: item.run.run_name,
                  values: item.series.transport_latency_ms,
                  color: pickColor(index),
                }))}
              />
              <LineChart
                title="Inference latency"
                xLabel="X 轴: step"
                series={compareItems.map((item, index) => ({
                  name: item.run.run_name,
                  values: item.series.inference_latency_ms,
                  color: pickColor(index),
                }))}
              />
              <LineChart
                title="Total latency"
                xLabel="X 轴: step"
                series={compareItems.map((item, index) => ({
                  name: item.run.run_name,
                  values: item.series.total_latency_ms,
                  color: pickColor(index),
                }))}
              />
            </>
          )}
        </section>
      ) : null}

      {activeTab === "distribution" ? (
        <section className="compare-grid">
          {compareItems.map((item, index) => (
            <article key={item.run.run_id} className="compare-card">
              <div className="run-detail-header">
                <div>
                  <span className="badge">{item.run.project}</span>
                  <h3>{item.run.run_name}</h3>
                </div>
                <span className={`status-chip tone-${scoreLatency(item.summary).tone}`}>
                  {scoreLatency(item.summary).label}
                </span>
              </div>
              <div className="timing-grid">
                <HistogramChart
                  title="Transport"
                  values={item.series.transport_latency_ms}
                  color={pickColor(index)}
                />
                <HistogramChart
                  title="Inference"
                  values={item.series.inference_latency_ms}
                  color={pickColor(index + 1)}
                />
                <HistogramChart
                  title="Total"
                  values={item.series.total_latency_ms}
                  color={pickColor(index + 2)}
                />
              </div>
            </article>
          ))}
          {!compareItems.length && !loadingCompare ? <div className="empty-panel">请选择至少一个 run。</div> : null}
        </section>
      ) : null}

      {activeTab === "detail" ? (
        <section className="section-stack">
          <section className="section-panel">
            <div className="table-shell">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Steps</th>
                    <th>Transport Avg / P95</th>
                    <th>Inference Avg / P95</th>
                    <th>Total Avg / P95 / Max</th>
                    <th>Assessment</th>
                  </tr>
                </thead>
                <tbody>
                  {compareItems.map((item) => {
                    const score = scoreLatency(item.summary);
                    return (
                      <tr key={item.run.run_id}>
                        <td>
                          <strong>{item.run.run_name}</strong>
                          <br />
                          <span className="muted">{item.run.project}</span>
                        </td>
                        <td>{item.series.steps?.length || 0}</td>
                        <td>
                          {formatMs(item.summary.transport_latency?.avg_ms)}
                          <br />
                          <span className="muted">P95 {formatMs(item.summary.transport_latency?.p95_ms)}</span>
                        </td>
                        <td>
                          {formatMs(item.summary.inference_latency?.avg_ms)}
                          <br />
                          <span className="muted">P95 {formatMs(item.summary.inference_latency?.p95_ms)}</span>
                        </td>
                        <td>
                          {formatMs(item.summary.total_latency?.avg_ms)}
                          <br />
                          <span className="muted">
                            P95 {formatMs(item.summary.total_latency?.p95_ms)} / Max {formatMs(item.summary.total_latency?.max_ms)}
                          </span>
                        </td>
                        <td>
                          <span className={`status-chip tone-${score.tone}`}>{score.label}</span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          <section className="detail-grid">
            {compareItems.map((item) => (
              <article key={item.run.run_id} className="section-panel">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">Assessment</p>
                    <h2>{item.run.run_name}</h2>
                  </div>
                </div>
                <div className="timing-grid">
                  {Object.entries(item.summary).map(([key, value]) => (
                    <div key={key} className="kpi-card">
                      <span className="stat-label">{metricLabel(key)}</span>
                      <strong>{formatMs(value?.avg_ms)}</strong>
                      <span className="muted">
                        P95 {formatMs(value?.p95_ms)} / Max {formatMs(value?.max_ms)}
                      </span>
                    </div>
                  ))}
                </div>
              </article>
            ))}
          </section>
        </section>
      ) : null}

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Inventory</p>
            <h2>完整 run 摘要</h2>
          </div>
          <span className="muted">
            {loadingRuns ? "加载 run 列表中..." : loadingCompare ? "更新对比中..." : "对比数据已同步"}
          </span>
        </div>
        <RunTable runs={filteredRuns} showProject />
      </section>
    </div>
  );
}
