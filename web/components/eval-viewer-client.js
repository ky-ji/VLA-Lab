"use client";

import { useEffect, useMemo, useState } from "react";

import { browserFetchJson, browserPostJson, toPublicApiUrl } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import {
  HeatmapChart,
  HistogramChart,
  LineChart,
  SimpleTabs,
  TrajectoryProjection,
} from "@/components/chart-kit";

function transpose(matrix) {
  if (!matrix?.length || !matrix[0]?.length) {
    return [];
  }
  return matrix[0].map((_, colIdx) => matrix.map((row) => row[colIdx] ?? null));
}

function perDimMetrics(gt, pred, labels) {
  if (!gt?.length || !pred?.length) {
    return [];
  }
  const dimCount = gt[0]?.length || 0;
  const rows = [];
  for (let dim = 0; dim < dimCount; dim += 1) {
    const errors = gt.map((row, idx) => (row?.[dim] ?? 0) - (pred[idx]?.[dim] ?? 0));
    const abs = errors.map((value) => Math.abs(value));
    rows.push({
      label: labels?.[dim] || `dim_${dim}`,
      mse: errors.reduce((sum, value) => sum + value * value, 0) / Math.max(errors.length, 1),
      mae: abs.reduce((sum, value) => sum + value, 0) / Math.max(abs.length, 1),
      max: Math.max(...abs),
      bias: errors.reduce((sum, value) => sum + value, 0) / Math.max(errors.length, 1),
      std: Math.sqrt(
        errors.reduce((sum, value) => sum + value * value, 0) / Math.max(errors.length, 1)
      ),
    });
  }
  return rows;
}

function inferDimCount(evalData) {
  return (
    evalData?.gt_actions?.[0]?.length ||
    evalData?.pred_actions?.[0]?.length ||
    evalData?.results?.action_keys?.length ||
    0
  );
}

export default function EvalViewerClient() {
  const [source, setSource] = useState("dir");
  const [dirPath, setDirPath] = useState("");
  const [resultsFile, setResultsFile] = useState("");
  const [trajId, setTrajId] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [activeTab, setActiveTab] = useState("timeseries");
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedDims, setSelectedDims] = useState([]);
  const [trajDims, setTrajDims] = useState({ x: 0, y: 1, z: 2 });

  async function loadEval(nextSource = source, nextDirPath = dirPath, nextResultsFile = resultsFile, nextTrajId = trajId) {
    setLoading(true);
    setError("");
    try {
      const data = await browserFetchJson("/api/eval/view", {
        source: nextSource,
        dir_path: nextSource === "dir" ? nextDirPath : undefined,
        results_file: nextSource === "dir" ? nextResultsFile || undefined : undefined,
        traj_id: nextTrajId === "" ? undefined : Number(nextTrajId),
      });
      setEvalData(data);
      const dims = inferDimCount(data);
      setSelectedDims(Array.from({ length: dims }, (_, idx) => idx));
      setTrajDims({
        x: 0,
        y: Math.min(1, Math.max(dims - 1, 0)),
        z: Math.min(2, Math.max(dims - 1, 0)),
      });
    } catch (err) {
      setError(String(err.message || err));
      setEvalData(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (source === "demo") {
      loadEval("demo", "", "", "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- loadEval is intentionally unstable
  }, [source]);

  async function handleUpload(file) {
    if (!file) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      const data = await browserPostJson("/api/eval/inline", payload);
      setSource("upload");
      setUploadedFileName(file.name);
      setEvalData(data);
      const dims = inferDimCount(data);
      setSelectedDims(Array.from({ length: dims }, (_, idx) => idx));
      setTrajDims({
        x: 0,
        y: Math.min(1, Math.max(dims - 1, 0)),
        z: Math.min(2, Math.max(dims - 1, 0)),
      });
    } catch (err) {
      setError(String(err.message || err));
      setEvalData(null);
    } finally {
      setLoading(false);
    }
  }

  const results = evalData?.results || {};
  const gtActions = evalData?.gt_actions || [];
  const predActions = evalData?.pred_actions || [];
  const dimCount = inferDimCount(evalData);
  const actionKeys = results.action_keys || Array.from({ length: dimCount }, (_, idx) => `dim_${idx}`);
  const dimMetrics = useMemo(() => perDimMetrics(gtActions, predActions, actionKeys), [gtActions, predActions, actionKeys]);
  const errorMatrix = transpose(
    gtActions.map((row, idx) => row.map((value, dim) => Math.abs(value - (predActions[idx]?.[dim] ?? 0))))
  );
  const selectedDimSet = new Set(selectedDims);

  return (
    <div className="section-stack">
      <section className="selection-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Eval Viewer</p>
            <h2>Open-loop 评估可视化</h2>
          </div>
        </div>
        <div className="selection-actions">
          <button type="button" className={source === "dir" ? "tab-pill is-active" : "tab-pill"} onClick={() => setSource("dir")}>
            浏览目录
          </button>
          <button type="button" className={source === "upload" ? "tab-pill is-active" : "tab-pill"} onClick={() => setSource("upload")}>
            上传 JSON
          </button>
          <button type="button" className={source === "demo" ? "tab-pill is-active" : "tab-pill"} onClick={() => setSource("demo")}>
            演示数据
          </button>
        </div>

        {source === "dir" ? (
          <div className="form-grid form-grid-wide">
            <input
              value={dirPath}
              onChange={(event) => setDirPath(event.target.value)}
              placeholder="/path/to/eval_outputs"
            />
            <button type="button" onClick={() => loadEval("dir", dirPath, resultsFile, trajId)}>
              {loading ? "加载中..." : "加载目录"}
            </button>
          </div>
        ) : null}

        {source === "upload" ? (
          <div className="selection-actions">
            <input
              type="file"
              accept=".json,application/json"
              onChange={(event) => handleUpload(event.target.files?.[0])}
            />
            <span className="muted">{uploadedFileName || "上传本地 results.json 进行查看"}</span>
          </div>
        ) : null}

        {source === "demo" ? (
          <div className="selection-actions">
            <button type="button" onClick={() => loadEval("demo", "", "", "")}>
              重新生成演示数据
            </button>
          </div>
        ) : null}

        {error ? <div className="placeholder-note">{error}</div> : null}
      </section>

      {evalData ? (
        <>
          <section className="section-panel">
            <div className="timing-grid">
              <div className="kpi-card">
                <span className="stat-label">Trajectories</span>
                <strong>{results.num_trajectories ?? 0}</strong>
              </div>
              <div className="kpi-card">
                <span className="stat-label">Avg MSE</span>
                <strong>{formatNumber(results.avg_mse, 6)}</strong>
              </div>
              <div className="kpi-card">
                <span className="stat-label">Avg MAE</span>
                <strong>{formatNumber(results.avg_mae, 6)}</strong>
              </div>
              <div className="kpi-card">
                <span className="stat-label">Action Horizon</span>
                <strong>{results.action_horizon ?? "--"}</strong>
              </div>
            </div>
          </section>

          {results?.results?.length ? (
            <section className="section-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Summary</p>
                  <h2>轨迹级评估摘要</h2>
                </div>
              </div>
              <div className="table-shell">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Trajectory</th>
                      <th>MSE</th>
                      <th>MAE</th>
                      <th>Steps</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.results.map((row, index) => (
                      <tr key={`${row.trajectory_id ?? index}`}>
                        <td>{row.trajectory_id ?? index}</td>
                        <td>{formatNumber(row.mse, 6)}</td>
                        <td>{formatNumber(row.mae, 6)}</td>
                        <td>{row.num_steps ?? "--"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}

          {source === "dir" ? (
            <section className="selection-panel">
              <div className="form-grid form-grid-wide">
                <select value={resultsFile} onChange={(event) => setResultsFile(event.target.value)}>
                  <option value="">选择结果文件</option>
                  {(evalData.json_files || []).map((fileName) => (
                    <option key={fileName} value={fileName}>
                      {fileName}
                    </option>
                  ))}
                </select>
                <select value={trajId} onChange={(event) => setTrajId(event.target.value)}>
                  <option value="">选择轨迹</option>
                  {(evalData.available_traj_ids || []).map((value) => (
                    <option key={value} value={value}>
                      Trajectory {value}
                    </option>
                  ))}
                </select>
                <button type="button" onClick={() => loadEval("dir", dirPath, resultsFile, trajId)}>
                  加载轨迹
                </button>
              </div>
            </section>
          ) : null}

          {gtActions.length && predActions.length ? (
            <>
              <section className="selection-panel">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">Dimensions</p>
                    <h2>选择显示维度</h2>
                  </div>
                </div>
                <div className="checkbox-grid">
                  {actionKeys.map((label, idx) => (
                    <label key={label} className="checkbox-item">
                      <input
                        type="checkbox"
                        checked={selectedDimSet.has(idx)}
                        onChange={(event) => {
                          setSelectedDims((current) =>
                            event.target.checked
                              ? [...current, idx].sort((a, b) => a - b)
                              : current.filter((value) => value !== idx)
                          );
                        }}
                      />
                      <span>{label}</span>
                    </label>
                  ))}
                </div>
                <SimpleTabs
                  activeTab={activeTab}
                  onChange={setActiveTab}
                  tabs={[
                    { key: "timeseries", label: "时序对比" },
                    { key: "heatmap", label: "误差热力图" },
                    { key: "metrics", label: "维度分析" },
                    { key: "trajectory", label: "3D 轨迹" },
                    { key: "distribution", label: "误差分布" },
                  ]}
                />
              </section>

              {activeTab === "timeseries" ? (
                <section className="section-stack">
                  {selectedDims.map((dimIdx) => (
                    <LineChart
                      key={dimIdx}
                      title={actionKeys[dimIdx]}
                      xLabel={`X 轴: trajectory step (action horizon ${results.action_horizon ?? "--"})`}
                      series={[
                        { name: "GT", values: gtActions.map((row) => row[dimIdx]), color: "#0f766e" },
                        { name: "Pred", values: predActions.map((row) => row[dimIdx]), color: "#dc2626" },
                      ]}
                    />
                  ))}
                </section>
              ) : null}

              {activeTab === "heatmap" ? (
                <section className="section-stack">
                  <HeatmapChart matrix={errorMatrix} rowLabels={actionKeys} title="Absolute error heatmap" />
                  <LineChart
                    title="Mean absolute error over time"
                    xLabel="X 轴: trajectory step"
                    series={[
                      {
                        name: "MAE",
                        values: gtActions.map((row, idx) =>
                          row.reduce((sum, value, dim) => sum + Math.abs(value - (predActions[idx]?.[dim] ?? 0)), 0) /
                          Math.max(row.length, 1)
                        ),
                        color: "#c2410c",
                      },
                    ]}
                  />
                </section>
              ) : null}

              {activeTab === "metrics" ? (
                <section className="section-panel">
                  <div className="table-shell">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Dim</th>
                          <th>MSE</th>
                          <th>MAE</th>
                          <th>Max</th>
                          <th>Bias</th>
                          <th>Std</th>
                        </tr>
                      </thead>
                      <tbody>
                        {dimMetrics.map((row) => (
                          <tr key={row.label}>
                            <td>{row.label}</td>
                            <td>{formatNumber(row.mse, 6)}</td>
                            <td>{formatNumber(row.mae, 6)}</td>
                            <td>{formatNumber(row.max, 6)}</td>
                            <td>{formatNumber(row.bias, 6)}</td>
                            <td>{formatNumber(row.std, 6)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              ) : null}

              {activeTab === "trajectory" ? (
                <section className="section-stack">
                  <section className="selection-panel">
                    <div className="control-grid">
                      <label>
                        <span>X 维度</span>
                        <select value={trajDims.x} onChange={(event) => setTrajDims((current) => ({ ...current, x: Number(event.target.value) }))}>
                          {actionKeys.map((label, idx) => (
                            <option key={`x-${label}`} value={idx}>
                              {label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        <span>Y 维度</span>
                        <select value={trajDims.y} onChange={(event) => setTrajDims((current) => ({ ...current, y: Number(event.target.value) }))}>
                          {actionKeys.map((label, idx) => (
                            <option key={`y-${label}`} value={idx}>
                              {label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        <span>Z 维度</span>
                        <select value={trajDims.z} onChange={(event) => setTrajDims((current) => ({ ...current, z: Number(event.target.value) }))}>
                          {actionKeys.map((label, idx) => (
                            <option key={`z-${label}`} value={idx}>
                              {label}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  </section>
                  <TrajectoryProjection
                    title="GT vs Pred trajectory"
                    series={[
                      {
                        name: "GT",
                        points: gtActions.map((row) => [row[trajDims.x], row[trajDims.y], row[trajDims.z]]),
                        color: "#0f766e",
                      },
                      {
                        name: "Pred",
                        points: predActions.map((row) => [row[trajDims.x], row[trajDims.y], row[trajDims.z]]),
                        color: "#dc2626",
                        dashed: true,
                      },
                    ]}
                  />
                </section>
              ) : null}

              {activeTab === "distribution" ? (
                <section className="detail-grid">
                  {selectedDims.map((dimIdx) => (
                    <HistogramChart
                      key={dimIdx}
                      title={`${actionKeys[dimIdx]} error`}
                      values={gtActions.map((row, idx) => (row[dimIdx] ?? 0) - (predActions[idx]?.[dimIdx] ?? 0))}
                      color={["#0f766e", "#c2410c", "#2563eb", "#7c3aed"][dimIdx % 4]}
                    />
                  ))}
                </section>
              ) : null}
            </>
          ) : (
            <section className="section-stack">
              <section className="placeholder-panel">
                <p>当前数据源没有加载到可交互的轨迹数组，可以先查看汇总指标或静态图。</p>
              </section>
              {evalData.static_pngs?.length ? (
                <section className="section-panel">
                  <div className="section-heading">
                    <div>
                      <p className="eyebrow">Static Plots</p>
                      <h2>回退到已保存的 PNG 图</h2>
                    </div>
                  </div>
                  <div className="gallery-grid">
                    {evalData.static_pngs.map((image) => (
                      <figure key={image.path}>
                        <img
                          src={toPublicApiUrl(`/api/eval/static-image?path=${encodeURIComponent(image.path)}`)}
                          alt={image.name}
                        />
                        <figcaption>{image.name}</figcaption>
                      </figure>
                    ))}
                  </div>
                </section>
              ) : null}
            </section>
          )}
        </>
      ) : null}
    </div>
  );
}
