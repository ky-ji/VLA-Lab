"use client";

import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";

import { browserFetchJson } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import { LineChart, TrajectoryProjection } from "@/components/chart-kit";

const DEFAULT_DATASET_PATH = "/data3/jikangye/vla-data/008_stack_bowls";

function pathName(path) {
  const parts = String(path || "").split("/").filter(Boolean);
  return parts[parts.length - 1] || "dataset";
}

function formatVector(values, limit = 8) {
  if (!Array.isArray(values) || values.length === 0) {
    return "--";
  }
  const visible = values.slice(0, limit).map((value) => formatNumber(value, 3));
  return `${visible.join(", ")}${values.length > limit ? " ..." : ""}`;
}

function episodeLengthFrom(info, episodeIdx) {
  const match = (info?.episode_lengths || []).find((item) => Number(item.episode_idx) === Number(episodeIdx));
  return Number(match?.length || 1);
}

export default function DatasetViewerClient() {
  const [datasetPath, setDatasetPath] = useState(DEFAULT_DATASET_PATH);
  const [loadedPath, setLoadedPath] = useState("");
  const [datasetCandidates, setDatasetCandidates] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [episodeView, setEpisodeView] = useState(null);
  const [episodeIdx, setEpisodeIdx] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);
  const [stepInterval, setStepInterval] = useState(5);
  const [maxFrames, setMaxFrames] = useState(20);
  const [workspaceRatio, setWorkspaceRatio] = useState(0.1);
  const [datasetMode, setDatasetMode] = useState("episode");
  const [isDatasetPlaying, setIsDatasetPlaying] = useState(false);
  const [datasetPlaybackSpeed, setDatasetPlaybackSpeed] = useState(2);
  const [loading, setLoading] = useState(false);
  const [episodeLoading, setEpisodeLoading] = useState(false);
  const [error, setError] = useState("");
  const datasetPlaybackTimerRef = useRef(null);
  const deferredStepIdx = useDeferredValue(stepIdx);

  async function loadDataset(path) {
    setIsDatasetPlaying(false);
    setLoading(true);
    setError("");
    try {
      const info = await browserFetchJson("/api/datasets/inspect", { path });
      setDatasetInfo(info);
      setLoadedPath(path);
      setEpisodeIdx(0);
      setStepIdx(0);
      setEpisodeView(null);
    } catch (err) {
      setError(String(err.message || err));
      setDatasetInfo(null);
      setEpisodeView(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let ignore = false;
    async function loadCandidates() {
      try {
        const payload = await browserFetchJson("/api/datasets/candidates");
        if (ignore) {
          return;
        }
        setDatasetCandidates(payload);
        const defaultPath = payload?.default_path || DEFAULT_DATASET_PATH;
        setDatasetPath(defaultPath);
        if (defaultPath) {
          loadDataset(defaultPath);
        }
      } catch (err) {
        if (!ignore) {
          setDatasetCandidates({ default_path: DEFAULT_DATASET_PATH, candidates: [] });
          setDatasetPath(DEFAULT_DATASET_PATH);
        }
      }
    }
    loadCandidates();
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    if (!datasetInfo || !loadedPath) {
      return;
    }
    let ignore = false;
    setEpisodeLoading(true);
    setError("");

    async function loadEpisode() {
      try {
        const view = await browserFetchJson("/api/datasets/episode", {
          path: loadedPath,
          episode_idx: episodeIdx,
          step_idx: deferredStepIdx,
          step_interval: stepInterval,
          max_frames: maxFrames,
          workspace_ratio: workspaceRatio,
        });
        if (!ignore) {
          setEpisodeView(view);
        }
      } catch (err) {
        if (!ignore) {
          setError(String(err.message || err));
        }
      } finally {
        if (!ignore) {
          setEpisodeLoading(false);
        }
      }
    }

    loadEpisode();
    return () => {
      ignore = true;
    };
  }, [datasetInfo, deferredStepIdx, episodeIdx, loadedPath, maxFrames, stepInterval, workspaceRatio]);

  const actions = episodeView?.actions || [];
  const currentAction = episodeView?.current_action || [];
  const currentState = episodeView?.current_state || [];
  const stepImages = episodeView?.step_images || {};
  const imageEntries = Object.entries(stepImages);
  const imageGrids = Object.entries(episodeView?.image_grids || {});
  const candidates = datasetCandidates?.candidates || [];
  const episodeLength = Number(episodeView?.episode_length || episodeLengthFrom(datasetInfo, episodeIdx));
  const maxStep = Math.max(0, episodeLength - 1);
  const displayStepIdx = Number(episodeView?.step_idx ?? Math.min(stepIdx, maxStep));
  const loadedName = pathName(loadedPath || datasetPath);
  const taskText = episodeView?.task || datasetInfo?.task || candidates.find((item) => item.path === loadedPath)?.task || "--";
  const modeButtons = [
    ["episode", "Episode Replay"],
    ["overview", "Dataset Overview"],
    ["workspace", "Workspace"],
  ];
  const datasetStats = [
    ["Format", datasetInfo?.format || "--"],
    ["Episodes", datasetInfo?.episode_count ?? "--"],
    ["Steps", datasetInfo?.total_steps ?? "--"],
    ["Images", datasetInfo?.image_keys?.length ?? "--"],
    ["Action", datasetInfo?.action_dim ?? "--"],
    ["State", datasetInfo?.state_dim ?? "--"],
  ];
  const signalCharts = useMemo(
    () => [
      {
        title: "XYZ action series",
        series: [
          { name: "x", values: episodeView?.xyz_series?.x || [], color: "#0f766e" },
          { name: "y", values: episodeView?.xyz_series?.y || [], color: "#c2410c" },
          { name: "z", values: episodeView?.xyz_series?.z || [], color: "#2563eb" },
        ],
      },
      {
        title: "State XYZ series",
        series: [
          { name: "x", values: episodeView?.state_xyz_series?.x || [], color: "#0f766e" },
          { name: "y", values: episodeView?.state_xyz_series?.y || [], color: "#c2410c" },
          { name: "z", values: episodeView?.state_xyz_series?.z || [], color: "#2563eb" },
        ],
      },
      {
        title: "Gripper series",
        series: [{ name: "gripper", values: episodeView?.gripper_series || [], color: "#16a34a" }],
      },
    ],
    [episodeView]
  );

  useEffect(() => {
    if (!isDatasetPlaying || !datasetInfo || maxStep <= 0) {
      if (datasetPlaybackTimerRef.current) {
        clearInterval(datasetPlaybackTimerRef.current);
        datasetPlaybackTimerRef.current = null;
      }
      return undefined;
    }

    const frameMs = 1000 / Math.max(0.25, Number(datasetPlaybackSpeed) || 1);
    datasetPlaybackTimerRef.current = setInterval(() => {
      if (!episodeLoading) {
        setStepIdx((current) => {
          return Math.min(maxStep, current + 1);
        });
      }
    }, frameMs);

    return () => {
      if (datasetPlaybackTimerRef.current) {
        clearInterval(datasetPlaybackTimerRef.current);
        datasetPlaybackTimerRef.current = null;
      }
    };
  }, [datasetInfo, datasetPlaybackSpeed, episodeLoading, isDatasetPlaying, maxStep]);

  useEffect(() => {
    if (isDatasetPlaying && stepIdx >= maxStep) {
      setIsDatasetPlaying(false);
    }
  }, [isDatasetPlaying, maxStep, stepIdx]);

  return (
    <div className="dataset-workspace-shell">
      <header className="dataset-workspace-header">
        <div className="dataset-workspace-title">
          <p className="eyebrow">Dataset Viewer</p>
          <h1>{loadedName}</h1>
          <p>{loadedPath || datasetPath}</p>
        </div>
        <div className="dataset-workspace-pathbar">
          <input
            value={datasetPath}
            onChange={(event) => setDatasetPath(event.target.value)}
            placeholder="输入 LeRobot / GR00T / Zarr 数据集路径"
          />
          <button type="button" onClick={() => loadDataset(datasetPath)} disabled={loading}>
            {loading ? "Loading" : "Load"}
          </button>
        </div>
      </header>

      {error ? <div className="dataset-workspace-error">{error}</div> : null}

      <div className="dataset-workspace-grid">
        <aside className="dataset-workspace-rail">
          <section className="dataset-workspace-panel dataset-workspace-mode-panel">
            {modeButtons.map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                className={`dataset-workspace-mode-button${datasetMode === mode ? " is-active" : ""}`}
                onClick={() => setDatasetMode(mode)}
              >
                {label}
              </button>
            ))}
          </section>

          <section className="dataset-workspace-panel">
            <div className="dataset-workspace-panel-head">
              <span>Default</span>
              <strong>{pathName(datasetCandidates?.default_path || DEFAULT_DATASET_PATH)}</strong>
            </div>
            <button
              type="button"
              className="dataset-workspace-default-button"
              onClick={() => {
                const defaultPath = datasetCandidates?.default_path || DEFAULT_DATASET_PATH;
                setDatasetPath(defaultPath);
                loadDataset(defaultPath);
              }}
            >
              {datasetCandidates?.default_path || DEFAULT_DATASET_PATH}
            </button>
          </section>

          <section className="dataset-workspace-panel dataset-workspace-candidates">
            <div className="dataset-workspace-panel-head">
              <span>Candidates</span>
              <strong>{candidates.length}</strong>
            </div>
            {candidates.map((candidate) => (
              <button
                key={candidate.path}
                type="button"
                className={`dataset-workspace-candidate-card dataset-candidate-card${
                  candidate.path === loadedPath ? " is-active" : ""
                }`}
                onClick={() => {
                  setDatasetPath(candidate.path);
                  loadDataset(candidate.path);
                }}
              >
                <span>{candidate.format || "dataset"}</span>
                <strong>{candidate.name || pathName(candidate.path)}</strong>
                <small>{candidate.episode_count} episodes · {candidate.total_steps} steps</small>
                <p>{candidate.task || candidate.path}</p>
              </button>
            ))}
          </section>

          {datasetInfo ? (
            <section className="dataset-workspace-panel dataset-workspace-episode-panel">
              <label>
                <span>Episode</span>
                <select
                  value={episodeIdx}
                  onChange={(event) => {
                    setIsDatasetPlaying(false);
                    setEpisodeIdx(Number(event.target.value));
                    setStepIdx(0);
                  }}
                >
                  {(datasetInfo.episode_lengths || []).map((item) => (
                    <option key={item.episode_idx} value={item.episode_idx}>
                      {item.episode_idx} · {item.length} steps
                    </option>
                  ))}
                </select>
              </label>
              <div className="dataset-workspace-stats">
                {datasetStats.map(([label, value]) => (
                  <div key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            </section>
          ) : null}
        </aside>

        <main className="dataset-workspace-viewer">
          {datasetInfo ? (
            <>
              <section className="dataset-workspace-panel dataset-workspace-stepbar">
                <div className="dataset-workspace-playback">
                  <button
                    type="button"
                    onClick={() => {
                      setIsDatasetPlaying(false);
                      setStepIdx(0);
                    }}
                  >
                    &lt;&lt;
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setIsDatasetPlaying(false);
                      setStepIdx(Math.max(0, stepIdx - 1));
                    }}
                  >
                    &lt;
                  </button>
                  <button
                    type="button"
                    className="dataset-workspace-play-button"
                    disabled={!datasetInfo || maxStep <= 0}
                    onClick={() => setIsDatasetPlaying((playing) => !playing)}
                  >
                    {isDatasetPlaying ? "Pause" : "Play"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setIsDatasetPlaying(false);
                      setStepIdx(Math.min(maxStep, stepIdx + 1));
                    }}
                  >
                    &gt;
                  </button>
                  <select
                    aria-label="Dataset playback speed"
                    value={datasetPlaybackSpeed}
                    onChange={(event) => setDatasetPlaybackSpeed(Number(event.target.value))}
                  >
                    <option value={0.5}>0.5x</option>
                    <option value={1}>1x</option>
                    <option value={2}>2x</option>
                    <option value={4}>4x</option>
                    <option value={8}>8x</option>
                  </select>
                </div>
                <input
                  type="range"
                  min="0"
                  max={maxStep}
                  value={Math.min(stepIdx, maxStep)}
                  onChange={(event) => {
                    setIsDatasetPlaying(false);
                    setStepIdx(Number(event.target.value));
                  }}
                />
                <div className="dataset-workspace-step-readout">
                  <span>Step</span>
                  <strong>{displayStepIdx} / {maxStep}</strong>
                  {episodeLoading && stepIdx !== displayStepIdx ? <small>Loading {Math.min(stepIdx, maxStep)}</small> : null}
                </div>
              </section>

              {datasetMode === "episode" ? (
                <section className="dataset-workspace-episode-layout">
                  <div className="dataset-workspace-panel dataset-workspace-frame-panel">
                    <div className="dataset-workspace-panel-head">
                      <span>Current Frames</span>
                      <strong>{imageEntries.length} cameras</strong>
                    </div>
                    {imageEntries.length ? (
                      <div className="dataset-workspace-image-grid step-gallery">
                        {imageEntries.map(([key, dataUrl]) => (
                          <figure key={key}>
                            <img src={dataUrl} alt={key} />
                            <figcaption>{key}</figcaption>
                          </figure>
                        ))}
                      </div>
                    ) : (
                      <div className="dataset-workspace-empty">No frame decoded for this step.</div>
                    )}
                  </div>
                  <div className="dataset-workspace-chart-grid">
                    {signalCharts.map((chart) => (
                      <LineChart
                        key={chart.title}
                        title={chart.title}
                        markerIndex={displayStepIdx}
                        xLabel="X axis: episode step"
                        series={chart.series}
                      />
                    ))}
                  </div>
                </section>
              ) : null}

              {datasetMode === "overview" ? (
                <section className="dataset-workspace-overview">
                  <div className="dataset-workspace-panel">
                    <div className="dataset-workspace-panel-head">
                      <span>Sampled Frames</span>
                      <strong>{maxFrames} max</strong>
                    </div>
                    {imageGrids.length ? (
                      imageGrids.map(([key, frames]) => (
                        <div key={key} className="dataset-workspace-gallery-section">
                          <h3>{key}</h3>
                          <div className="dataset-workspace-sample-grid gallery-grid">
                            {frames.map((frame) => (
                              <figure key={`${key}-${frame.step_idx}`}>
                                <img src={frame.data_url} alt={`${key}-${frame.step_idx}`} />
                                <figcaption>Step {frame.step_idx}</figcaption>
                              </figure>
                            ))}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="dataset-workspace-empty">No sampled frames available.</div>
                    )}
                  </div>
                  <div className="dataset-workspace-chart-grid">
                    {signalCharts.map((chart) => (
                      <LineChart
                        key={chart.title}
                        title={chart.title}
                        markerIndex={displayStepIdx}
                        xLabel="X axis: episode step"
                        series={chart.series}
                      />
                    ))}
                  </div>
                </section>
              ) : null}

              {datasetMode === "workspace" ? (
                <section className="dataset-workspace-panel dataset-workspace-trajectory-panel">
                  <TrajectoryProjection
                    title="Workspace samples"
                    currentPoint={currentAction}
                    series={[
                      {
                        name: "Workspace",
                        points: episodeView?.workspace_points || [],
                        color: "#7c3aed",
                        mode: "markers",
                        markerSize: 3,
                        opacity: 0.45,
                      },
                      { name: "Action path", points: actions, color: "#0f766e" },
                    ]}
                  />
                </section>
              ) : null}
            </>
          ) : (
            <section className="dataset-workspace-panel dataset-workspace-empty-state">
              <p className="eyebrow">No Dataset Loaded</p>
              <h2>选择候选路径或输入数据集路径</h2>
              <p>默认会扫描 /data3/jikangye/vla-data，并优先加载 008_stack_bowls。</p>
            </section>
          )}
        </main>

        <aside className="dataset-workspace-inspector">
          <section className="dataset-workspace-panel">
            <div className="dataset-workspace-panel-head">
              <span>Current Step</span>
              <strong>{displayStepIdx}</strong>
            </div>
            <div className="dataset-workspace-current-values">
              <div>
                <span>Current Action</span>
                <p>{formatVector(currentAction)}</p>
              </div>
              <div>
                <span>Current State</span>
                <p>{formatVector(currentState)}</p>
              </div>
              <div>
                <span>Task</span>
                <p>{taskText}</p>
              </div>
            </div>
          </section>

          <section className="dataset-workspace-panel dataset-workspace-controls-panel">
            <label>
              <span>Sampling interval</span>
              <input
                type="number"
                min="1"
                max="20"
                value={stepInterval}
                onChange={(event) => setStepInterval(Number(event.target.value))}
              />
            </label>
            <label>
              <span>Max frames</span>
              <input
                type="number"
                min="5"
                max="50"
                value={maxFrames}
                onChange={(event) => setMaxFrames(Number(event.target.value))}
              />
            </label>
            <label>
              <span>Workspace ratio</span>
              <input
                type="number"
                min="0.01"
                max="1"
                step="0.01"
                value={workspaceRatio}
                onChange={(event) => setWorkspaceRatio(Number(event.target.value))}
              />
            </label>
          </section>

          <section className="dataset-workspace-panel">
            <details>
              <summary>Low-dim JSON</summary>
              <pre>{JSON.stringify(episodeView?.lowdim_step || {}, null, 2)}</pre>
            </details>
          </section>
        </aside>
      </div>
    </div>
  );
}
