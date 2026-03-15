import Link from "next/link";

import RunCard from "@/components/run-card";
import { getOverview } from "@/lib/api";
import { formatDate, formatShortText } from "@/lib/format";

export const dynamic = "force-dynamic";

function runHref(run) {
  return `/runs/${encodeURIComponent(run.project)}/${encodeURIComponent(run.run_name)}`;
}

export default async function HomePage() {
  const overview = await getOverview();
  const latestRuns = overview?.latest_runs ?? [];
  const latestRun = latestRuns[0] || null;

  return (
    <div className="section-stack">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Workspace Overview</p>
          <h1>把常用排查入口放在一个更稳、更快的工作台里。</h1>
          <p>
            这里优先展示你最常看的状态：工作目录、最近运行、项目规模，以及 Runs / Latency / Dataset / Eval
            四条主链路，减少来回翻页找入口的成本。
          </p>
          <div className="hero-actions">
            <Link href="/runs" className="button-link">
              打开 Runs
            </Link>
            <Link href="/latency" className="button-ghost">
              查看 Latency
            </Link>
          </div>
          <div className="feature-grid">
            <div className="feature-card">
              <h3>Runs 详情</h3>
              <p>按项目浏览 run，进入 step 回放、attention 和动作轨迹分析。</p>
            </div>
            <div className="feature-card">
              <h3>数据与评测</h3>
              <p>Dataset workspace、open-loop eval 和可视化入口都收进同一套导航。</p>
            </div>
          </div>
        </div>
        <aside className="hero-aside">
          <div className="metric-grid">
            <div className="metric-card">
              <span className="stat-label">Runs Directory</span>
              <div className="path-text" title={overview?.runs_dir ?? "--"}>
                {overview?.runs_dir ?? "--"}
              </div>
            </div>
            <div className="metric-card">
              <span className="stat-label">Projects</span>
              <strong>{overview?.project_count ?? 0}</strong>
            </div>
            <div className="metric-card">
              <span className="stat-label">Runs</span>
              <strong>{overview?.run_count ?? 0}</strong>
            </div>
          </div>
          <div className="section-panel compact-panel overview-highlight">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Latest Run</p>
                <h2>{latestRun ? latestRun.run_name : "暂无运行记录"}</h2>
              </div>
              {latestRun ? (
                <Link href={runHref(latestRun)} className="text-link">
                  打开详情
                </Link>
              ) : null}
            </div>
            {latestRun ? (
              <div className="overview-list">
                <div className="overview-row">
                  <span className="stat-label">Project</span>
                  <strong>{latestRun.project}</strong>
                </div>
                <div className="overview-row">
                  <span className="stat-label">Model</span>
                  <span title={latestRun.model_name || "unknown"}>{formatShortText(latestRun.model_name || "unknown", 48)}</span>
                </div>
                <div className="overview-row">
                  <span className="stat-label">Task</span>
                  <span title={latestRun.task_name || "No task"}>{formatShortText(latestRun.task_name || "No task", 72)}</span>
                </div>
                <div className="overview-row">
                  <span className="stat-label">Updated</span>
                  <span>{formatDate(latestRun.updated_at)}</span>
                </div>
              </div>
            ) : (
              <p className="muted">当前 runs 目录里还没有可展示的运行结果。</p>
            )}
          </div>
        </aside>
      </section>

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Quick Access</p>
            <h2>常用入口</h2>
          </div>
        </div>
        <div className="feature-grid">
          <div className="feature-card">
            <h3>Inference Runs</h3>
            <p>进入项目和 run 列表，快速切到具体 step、图像、轨迹和 attention 工作区。</p>
            <Link href="/runs" className="text-link">
              进入 Runs
            </Link>
          </div>
          <div className="feature-card">
            <h3>Latency Compare</h3>
            <p>挑多个 run 做 transport / inference / total latency 的对比和分布分析。</p>
            <Link href="/latency" className="text-link">
              打开 Latency
            </Link>
          </div>
          <div className="feature-card">
            <h3>Dataset Workspace</h3>
            <p>查看 episode 图像网格、动作序列和工作空间点云，不再整页 rerun。</p>
            <Link href="/datasets" className="text-link">
              打开 Datasets
            </Link>
          </div>
          <div className="feature-card">
            <h3>Eval Viewer</h3>
            <p>读取目录或 JSON 结果，检查预测轨迹、误差分布和静态可视化图片。</p>
            <Link href="/eval" className="text-link">
              打开 Eval
            </Link>
          </div>
        </div>
      </section>

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Latest</p>
            <h2>最近的运行记录</h2>
          </div>
          <Link href="/runs" className="text-link">
            See all runs
          </Link>
        </div>
        <div className="run-card-grid">
          {latestRuns.map((run) => (
            <RunCard key={run.run_id} run={run} />
          ))}
        </div>
      </section>
    </div>
  );
}
