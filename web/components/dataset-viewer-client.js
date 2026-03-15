"use client";

import { useDeferredValue, useEffect, useState } from "react";

import { browserFetchJson } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import { LineChart, SimpleTabs, TrajectoryProjection } from "@/components/chart-kit";

const DEFAULT_DATASET_PATH =
  "/data0/vla-data/processed/Diffusion_Policy/data/001_assembly_chocolate/assembly_chocolate_300.zarr";

export default function DatasetViewerClient() {
  const [datasetPath, setDatasetPath] = useState(DEFAULT_DATASET_PATH);
  const [loadedPath, setLoadedPath] = useState("");
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [episodeView, setEpisodeView] = useState(null);
  const [episodeIdx, setEpisodeIdx] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);
  const [stepInterval, setStepInterval] = useState(5);
  const [maxFrames, setMaxFrames] = useState(20);
  const [workspaceRatio, setWorkspaceRatio] = useState(0.1);
  const [activeTab, setActiveTab] = useState("detail");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const deferredStepIdx = useDeferredValue(stepIdx);

  async function loadDataset(path) {
    setLoading(true);
    setError("");
    try {
      const info = await browserFetchJson("/api/datasets/inspect", { path });
      setDatasetInfo(info);
      setLoadedPath(path);
      setEpisodeIdx(0);
      setStepIdx(0);
    } catch (err) {
      setError(String(err.message || err));
      setDatasetInfo(null);
      setEpisodeView(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!datasetInfo || !loadedPath) {
      return;
    }
    let ignore = false;
    setLoading(true);
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
          setLoading(false);
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
  const stepImages = episodeView?.step_images || {};

  return (
    <div className="section-stack">
      <section className="selection-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Dataset Viewer</p>
            <h2>Zarr 数据集浏览</h2>
          </div>
        </div>
        <div className="form-grid">
          <input
            value={datasetPath}
            onChange={(event) => setDatasetPath(event.target.value)}
            placeholder="输入 zarr 数据集路径"
          />
          <button type="button" onClick={() => loadDataset(datasetPath)}>
            {loading ? "加载中..." : "加载数据集"}
          </button>
        </div>
        {error ? <div className="placeholder-note">{error}</div> : null}
      </section>

      {datasetInfo ? (
        <>
          <section className="section-panel">
            <div className="timing-grid">
              <div className="kpi-card">
                <span className="stat-label">Episodes</span>
                <strong>{datasetInfo.episode_count}</strong>
              </div>
              <div className="kpi-card">
                <span className="stat-label">Total Steps</span>
                <strong>{datasetInfo.total_steps}</strong>
              </div>
              <div className="kpi-card">
                <span className="stat-label">Image Keys</span>
                <strong>{datasetInfo.image_keys.length}</strong>
              </div>
              <div className="kpi-card">
                <span className="stat-label">Action Dim</span>
                <strong>{datasetInfo.action_dim}</strong>
              </div>
            </div>
          </section>

          <section className="selection-panel">
            <div className="form-grid form-grid-wide">
              <select value={episodeIdx} onChange={(event) => setEpisodeIdx(Number(event.target.value))}>
                {(datasetInfo.episode_lengths || []).map((item) => (
                  <option key={item.episode_idx} value={item.episode_idx}>
                    Episode {item.episode_idx} ({item.length} steps)
                  </option>
                ))}
              </select>
              <input
                type="range"
                min="0"
                max={Math.max(0, (episodeView?.episode_length || 1) - 1)}
                value={stepIdx}
                onChange={(event) => setStepIdx(Number(event.target.value))}
              />
              <span className="range-readout">Step {stepIdx}</span>
            </div>
            <div className="control-grid">
              <label>
                <span>采样间隔</span>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={stepInterval}
                  onChange={(event) => setStepInterval(Number(event.target.value))}
                />
              </label>
              <label>
                <span>最大帧数</span>
                <input
                  type="number"
                  min="5"
                  max="50"
                  value={maxFrames}
                  onChange={(event) => setMaxFrames(Number(event.target.value))}
                />
              </label>
              <label>
                <span>空间采样比例</span>
                <input
                  type="number"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={workspaceRatio}
                  onChange={(event) => setWorkspaceRatio(Number(event.target.value))}
                />
              </label>
            </div>
            <SimpleTabs
              activeTab={activeTab}
              onChange={setActiveTab}
              tabs={[
                { key: "detail", label: "详细交互" },
                { key: "overview", label: "全局概览" },
                { key: "workspace", label: "空间分布" },
              ]}
            />
          </section>

          {activeTab === "detail" ? (
            <section className="detail-grid">
              <div className="section-panel">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">Step Images</p>
                    <h2>逐帧观测</h2>
                  </div>
                </div>
                <div className="step-gallery">
                  {Object.entries(stepImages).map(([key, dataUrl]) => (
                    <figure key={key}>
                      <img src={dataUrl} alt={key} />
                      <figcaption>{key}</figcaption>
                    </figure>
                  ))}
                </div>
              </div>
              <div className="section-stack">
                <div className="section-panel">
                  <div className="section-heading">
                    <div>
                      <p className="eyebrow">Current Step</p>
                      <h2>动作与状态</h2>
                    </div>
                  </div>
                  <div className="meta-panel">
                    <div>
                      <span className="stat-label">Current Action</span>
                      <p>{currentAction.map((value) => formatNumber(value, 3)).join(", ") || "--"}</p>
                    </div>
                    <div>
                      <span className="stat-label">Low-dim State</span>
                      <pre>{JSON.stringify(episodeView?.lowdim_step || {}, null, 2)}</pre>
                    </div>
                  </div>
                </div>
                <TrajectoryProjection
                  title="Episode trajectory"
                  currentPoint={currentAction}
                  series={[{ name: "Action path", points: actions, color: "#0f766e" }]}
                />
              </div>
            </section>
          ) : null}

          {activeTab === "overview" ? (
            <section className="section-stack">
              <section className="section-panel">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">Frames</p>
                    <h2>采样图像网格</h2>
                  </div>
                </div>
                {Object.entries(episodeView?.image_grids || {}).map(([key, frames]) => (
                  <div key={key} className="gallery-section">
                    <h3>{key}</h3>
                    <div className="gallery-grid">
                      {frames.map((frame) => (
                        <figure key={`${key}-${frame.step_idx}`}>
                          <img src={frame.data_url} alt={`${key}-${frame.step_idx}`} />
                          <figcaption>Step {frame.step_idx}</figcaption>
                        </figure>
                      ))}
                    </div>
                  </div>
                ))}
              </section>
              <section className="detail-grid">
                <LineChart
                  title="XYZ action series"
                  markerIndex={stepIdx}
                  xLabel="X 轴: episode step"
                  series={[
                    { name: "x", values: episodeView?.xyz_series?.x || [], color: "#0f766e" },
                    { name: "y", values: episodeView?.xyz_series?.y || [], color: "#c2410c" },
                    { name: "z", values: episodeView?.xyz_series?.z || [], color: "#2563eb" },
                  ]}
                />
                <LineChart
                  title="Gripper series"
                  markerIndex={stepIdx}
                  xLabel="X 轴: episode step"
                  series={[{ name: "gripper", values: episodeView?.gripper_series || [], color: "#16a34a" }]}
                />
              </section>
            </section>
          ) : null}

          {activeTab === "workspace" ? (
            <section className="section-panel">
              <TrajectoryProjection
                title="Workspace samples"
                series={[
                  {
                    name: "Workspace",
                    points: episodeView?.workspace_points || [],
                    color: "#7c3aed",
                    mode: "markers",
                    markerSize: 3,
                    opacity: 0.45,
                  },
                ]}
              />
            </section>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
