"use client";

import { useEffect, useMemo, useState } from "react";

import { browserFetchJson, browserPostJson, toPublicApiUrl } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import {
  HeatmapChart,
  HistogramChart,
  LineChart,
  TrajectoryProjection,
} from "@/components/chart-kit";

const DEFAULT_EVAL_PATH = "/data3/jikangye/workspace/baselines/vla-baselines/Isaac-GR00T/outputs/vlalab_eval";

const EVAL_MODES = [
  ["timeseries", "Timeseries"],
  ["heatmap", "Error Map"],
  ["metrics", "Metrics"],
  ["trajectory", "Trajectory"],
  ["plots", "Plots"],
];

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

function inferActionLabels(results, dimCount) {
  const keys = results?.action_keys || [];
  if (keys.length === dimCount) {
    return keys;
  }
  if (keys.includes("robot_eef_pose") && dimCount >= 7) {
    return ["eef_x", "eef_y", "eef_z", "eef_rx", "eef_ry", "eef_rz", "gripper"].slice(0, dimCount);
  }
  return Array.from({ length: dimCount }, (_, idx) => keys[idx] || `dim_${idx}`);
}

function pathName(path) {
  if (!path) {
    return "--";
  }
  return String(path).split("/").filter(Boolean).pop() || path;
}

function formatVector(values, limit = 7) {
  if (!values?.length) {
    return "--";
  }
  return values
    .slice(0, limit)
    .map((value) => formatNumber(value, 4))
    .join(", ");
}

export default function EvalViewerClient() {
  const [source, setSource] = useState("dir");
  const [dirPath, setDirPath] = useState(DEFAULT_EVAL_PATH);
  const [resultsFile, setResultsFile] = useState("");
  const [trajId, setTrajId] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [activeTab, setActiveTab] = useState("timeseries");
  const [evalCandidates, setEvalCandidates] = useState(null);
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
      if (nextSource === "dir") {
        setDirPath(data.dir_path || nextDirPath);
        setResultsFile(data.selected_json || "");
        setTrajId(data.selected_traj_id === null || data.selected_traj_id === undefined ? "" : String(data.selected_traj_id));
      }
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
    let ignore = false;
    async function loadCandidates() {
      try {
        const payload = await browserFetchJson("/api/eval/candidates");
        if (ignore) {
          return;
        }
        setEvalCandidates(payload);
        const defaultPath = payload?.default_path || DEFAULT_EVAL_PATH;
        setDirPath(defaultPath);
        loadEval("dir", defaultPath, "", "");
      } catch (err) {
        if (ignore) {
          return;
        }
        setEvalCandidates({ default_path: DEFAULT_EVAL_PATH, candidates: [] });
        setError(String(err.message || err));
      }
    }
    loadCandidates();
    return () => {
      ignore = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- initial candidate bootstrap only
  }, []);

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
  const staticPngs = evalData?.static_pngs || [];
  const dimCount = inferDimCount(evalData);
  const actionKeys = inferActionLabels(results, dimCount);
  const dimMetrics = useMemo(() => perDimMetrics(gtActions, predActions, actionKeys), [gtActions, predActions, actionKeys]);
  const selectedDimSet = new Set(selectedDims);
  const visibleDims = selectedDims.filter((idx) => idx >= 0 && idx < dimCount);
  const errorMatrix = transpose(
    gtActions.map((row, idx) => row.map((value, dim) => Math.abs(value - (predActions[idx]?.[dim] ?? 0))))
  );
  const meanAbsoluteErrorSeries = gtActions.map((row, idx) =>
    row.reduce((sum, value, dim) => sum + Math.abs(value - (predActions[idx]?.[dim] ?? 0)), 0) / Math.max(row.length, 1)
  );
  const defaultPath = evalCandidates?.default_path || DEFAULT_EVAL_PATH;
  const candidateList = evalCandidates?.candidates || [];
  const trajectoryRows = Array.isArray(results.results) ? results.results : [];
  const selectedTrajectoryRow =
    trajectoryRows.find((row) => String(row.trajectory_id) === String(trajId)) || trajectoryRows[0] || null;
  const hasSeries = gtActions.length > 0 && predActions.length > 0;
  const selectedTrajectoryLabel = evalData?.selected_traj_id ?? selectedTrajectoryRow?.trajectory_id ?? "--";
  const metricCards = [
    ["Trajectories", results.num_trajectories ?? trajectoryRows.length ?? 0],
    ["Avg MSE", formatNumber(results.avg_mse, 6)],
    ["Avg MAE", formatNumber(results.avg_mae, 6)],
    ["Horizon", results.action_horizon ?? "--"],
  ];

  return (
    <div className="eval-workspace-shell">
      <header className="eval-workspace-header">
        <div className="eval-workspace-title">
          <p className="eyebrow">Eval Viewer</p>
          <h1>Open-loop 评估可视化</h1>
          <p>{source === "dir" ? dirPath : uploadedFileName || "Demo / uploaded eval payload"}</p>
        </div>
        <div className="eval-workspace-pathbar">
          <input
            value={dirPath}
            onChange={(event) => setDirPath(event.target.value)}
            placeholder={defaultPath}
          />
          <button
            type="button"
            onClick={() => {
              setSource("dir");
              loadEval("dir", dirPath, resultsFile, trajId);
            }}
            disabled={loading}
          >
            {loading ? "Loading" : "Load"}
          </button>
        </div>
      </header>

      {error ? <div className="eval-workspace-error">{error}</div> : null}

      <div className="eval-workspace-grid">
        <aside className="eval-workspace-rail">
          <section className="eval-workspace-panel eval-workspace-mode-panel">
            {EVAL_MODES.map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                className={`eval-workspace-mode-button${activeTab === mode ? " is-active" : ""}`}
                onClick={() => setActiveTab(mode)}
              >
                {label}
              </button>
            ))}
          </section>

          <section className="eval-workspace-panel eval-workspace-source-panel">
            <div className="eval-workspace-panel-head">
              <span>Source</span>
              <strong>{source}</strong>
            </div>
            <div className="eval-workspace-source-buttons">
              {[
                ["dir", "Directory"],
                ["upload", "Upload"],
                ["demo", "Demo"],
              ].map(([mode, label]) => (
                <button
                  key={mode}
                  type="button"
                  className={source === mode ? "is-active" : ""}
                  onClick={() => {
                    setSource(mode);
                    if (mode === "dir") {
                      loadEval("dir", dirPath || defaultPath, resultsFile, trajId);
                    }
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
            {source === "upload" ? (
              <label className="eval-workspace-file-input">
                <span>Results JSON</span>
                <input
                  type="file"
                  accept=".json,application/json"
                  onChange={(event) => handleUpload(event.target.files?.[0])}
                />
              </label>
            ) : null}
            {source === "demo" ? (
              <button type="button" className="eval-workspace-default-button" onClick={() => loadEval("demo", "", "", "")}>
                重新生成演示数据
              </button>
            ) : null}
          </section>

          {source === "dir" ? (
            <>
              <section className="eval-workspace-panel">
                <div className="eval-workspace-panel-head">
                  <span>Default</span>
                  <strong>{pathName(defaultPath)}</strong>
                </div>
                <button
                  type="button"
                  className="eval-workspace-default-button eval-candidate-card"
                  onClick={() => {
                    setDirPath(defaultPath);
                    setResultsFile("");
                    setTrajId("");
                    loadEval("dir", defaultPath, "", "");
                  }}
                >
                  {defaultPath}
                </button>
              </section>

              <section className="eval-workspace-panel eval-workspace-candidates">
                <div className="eval-workspace-panel-head">
                  <span>Candidates</span>
                  <strong>{candidateList.length}</strong>
                </div>
                <div className="eval-candidate-grid">
                  {candidateList.map((candidate) => (
                    <button
                      type="button"
                      key={candidate.path}
                      className={`eval-workspace-candidate-card eval-candidate-card dataset-candidate-card${
                        dirPath === candidate.path ? " is-active" : ""
                      }`}
                      onClick={() => {
                        setDirPath(candidate.path);
                        setResultsFile("");
                        setTrajId("");
                        loadEval("dir", candidate.path, "", "");
                      }}
                    >
                      <span>{candidate.format || "openloop"}</span>
                      <strong>{candidate.name || pathName(candidate.path)}</strong>
                      <small>
                        {candidate.json_count ?? 0} json · {candidate.trajectory_count ?? 0} traj · {candidate.png_count ?? 0} plots
                      </small>
                      <p>{candidate.path}</p>
                    </button>
                  ))}
                </div>
              </section>

              {evalData ? (
                <section className="eval-workspace-panel eval-workspace-trajectory-panel">
                  <label>
                    <span>Result file</span>
                    <select value={resultsFile} onChange={(event) => setResultsFile(event.target.value)}>
                      <option value="">选择结果文件</option>
                      {(evalData.json_files || []).map((fileName) => (
                        <option key={fileName} value={fileName}>
                          {fileName}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    <span>Trajectory</span>
                    <select value={trajId} onChange={(event) => setTrajId(event.target.value)}>
                      <option value="">选择轨迹</option>
                      {(evalData.available_traj_ids || []).map((value) => (
                        <option key={value} value={value}>
                          Trajectory {value}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button type="button" onClick={() => loadEval("dir", dirPath, resultsFile, trajId)}>
                    加载轨迹
                  </button>
                </section>
              ) : null}
            </>
          ) : null}
        </aside>

        <main className="eval-workspace-viewer">
          {evalData ? (
            <>
              <section className="eval-workspace-panel eval-workspace-viewer-head">
                <div>
                  <span>Trajectory</span>
                  <strong>{selectedTrajectoryLabel}</strong>
                </div>
                <div>
                  <span>Steps</span>
                  <strong>{gtActions.length || selectedTrajectoryRow?.num_steps || "--"}</strong>
                </div>
                <div>
                  <span>Mode</span>
                  <strong>{EVAL_MODES.find(([mode]) => mode === activeTab)?.[1]}</strong>
                </div>
              </section>

              {activeTab === "timeseries" ? (
                <section className="eval-workspace-chart-grid">
                  {hasSeries && visibleDims.length ? (
                    visibleDims.map((dimIdx) => (
                      <LineChart
                        key={dimIdx}
                        title={actionKeys[dimIdx]}
                        xLabel={`X axis: trajectory step (${gtActions.length} steps)`}
                        series={[
                          { name: "GT", values: gtActions.map((row) => row[dimIdx]), color: "#0f766e" },
                          { name: "Pred", values: predActions.map((row) => row[dimIdx]), color: "#dc2626" },
                        ]}
                      />
                    ))
                  ) : (
                    <section className="eval-workspace-panel eval-workspace-empty">No trajectory arrays loaded.</section>
                  )}
                </section>
              ) : null}

              {activeTab === "heatmap" ? (
                <section className="eval-workspace-analysis-grid">
                  {hasSeries ? (
                    <>
                      <HeatmapChart matrix={errorMatrix} rowLabels={actionKeys} title="Absolute error heatmap" />
                      <LineChart
                        title="Mean absolute error over time"
                        xLabel="X axis: trajectory step"
                        series={[{ name: "MAE", values: meanAbsoluteErrorSeries, color: "#c2410c" }]}
                      />
                    </>
                  ) : (
                    <section className="eval-workspace-panel eval-workspace-empty">No trajectory arrays loaded.</section>
                  )}
                </section>
              ) : null}

              {activeTab === "metrics" ? (
                <section className="eval-workspace-panel">
                  <div className="eval-workspace-panel-head">
                    <span>Dimension Metrics</span>
                    <strong>{dimMetrics.length}</strong>
                  </div>
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
                <section className="eval-workspace-panel eval-workspace-trajectory-view">
                  <div className="eval-workspace-axis-controls">
                    {[
                      ["x", "X dim"],
                      ["y", "Y dim"],
                      ["z", "Z dim"],
                    ].map(([axis, label]) => (
                      <label key={axis}>
                        <span>{label}</span>
                        <select value={trajDims[axis]} onChange={(event) => setTrajDims((current) => ({ ...current, [axis]: Number(event.target.value) }))}>
                          {actionKeys.map((labelText, idx) => (
                            <option key={`${axis}-${labelText}`} value={idx}>
                              {labelText}
                            </option>
                          ))}
                        </select>
                      </label>
                    ))}
                  </div>
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

              {activeTab === "plots" ? (
                <section className="eval-workspace-panel">
                  <div className="eval-workspace-panel-head">
                    <span>Static Plots</span>
                    <strong>{staticPngs.length}</strong>
                  </div>
                  {staticPngs.length ? (
                    <div className="eval-workspace-static-grid gallery-grid">
                      {staticPngs.map((image) => (
                        <figure key={image.path}>
                          <img
                            src={toPublicApiUrl(`/api/eval/static-image?path=${encodeURIComponent(image.path)}&dir_path=${encodeURIComponent(evalData.dir_path || "")}`)}
                            alt={image.name}
                          />
                          <figcaption>{image.name}</figcaption>
                        </figure>
                      ))}
                    </div>
                  ) : (
                    <div className="eval-workspace-empty">No saved PNG plots in this eval folder.</div>
                  )}
                </section>
              ) : null}
            </>
          ) : (
            <section className="eval-workspace-panel eval-workspace-empty-state">
              <p className="eyebrow">No Eval Loaded</p>
              <h2>选择候选目录或上传 results.json</h2>
              <p>默认会优先加载 Isaac-GR00T 的 vlalab_eval 输出。</p>
            </section>
          )}
        </main>

        <aside className="eval-workspace-inspector">
          <section className="eval-workspace-panel">
            <div className="eval-workspace-panel-head">
              <span>Overview</span>
              <strong>{pathName(evalData?.dir_path || uploadedFileName || "eval")}</strong>
            </div>
            <div className="eval-workspace-metric-grid">
              {metricCards.map(([label, value]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </section>

          {selectedTrajectoryRow ? (
            <section className="eval-workspace-panel">
              <div className="eval-workspace-panel-head">
                <span>Selected</span>
                <strong>{selectedTrajectoryLabel}</strong>
              </div>
              <div className="eval-workspace-current-values">
                <div>
                  <span>MSE / MAE</span>
                  <p>{formatNumber(selectedTrajectoryRow.mse, 6)} / {formatNumber(selectedTrajectoryRow.mae, 6)}</p>
                </div>
                <div>
                  <span>GT first</span>
                  <p>{formatVector(gtActions[0])}</p>
                </div>
                <div>
                  <span>Pred first</span>
                  <p>{formatVector(predActions[0])}</p>
                </div>
              </div>
            </section>
          ) : null}

          {trajectoryRows.length ? (
            <section className="eval-workspace-panel">
              <div className="eval-workspace-panel-head">
                <span>Summary</span>
                <strong>{trajectoryRows.length}</strong>
              </div>
              <div className="eval-workspace-summary-table">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Traj</th>
                      <th>MSE</th>
                      <th>Steps</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trajectoryRows.map((row, index) => (
                      <tr key={`${row.trajectory_id ?? index}`}>
                        <td>{row.trajectory_id ?? index}</td>
                        <td>{formatNumber(row.mse, 5)}</td>
                        <td>{row.num_steps ?? "--"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}

          {hasSeries ? (
            <section className="eval-workspace-panel">
              <div className="eval-workspace-panel-head">
                <span>Dimensions</span>
                <strong>{visibleDims.length} / {dimCount}</strong>
              </div>
              <div className="eval-workspace-dim-grid">
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
              <div className="eval-workspace-dim-table">
                {dimMetrics.slice(0, 8).map((row) => (
                  <div key={row.label}>
                    <span>{row.label}</span>
                    <strong>{formatNumber(row.mae, 5)}</strong>
                  </div>
                ))}
              </div>
            </section>
          ) : null}

          <section className="eval-workspace-panel">
            <details>
              <summary>Raw results JSON</summary>
              <pre>{JSON.stringify(results || {}, null, 2)}</pre>
            </details>
          </section>
        </aside>
      </div>
    </div>
  );
}
