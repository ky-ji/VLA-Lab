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
          <p className="eyebrow">VLA-Lab Workspace</p>
          <h1>VLA 部署<br />运维工作台</h1>
          <p className="hero-desc">
            统一管理推理日志、时延分析、数据集浏览与开环评估，减少排查成本。
          </p>
          <div className="hero-actions">
            <Link href="/runs" className="button-link">
              浏览 Runs
            </Link>
            <Link href="/latency" className="button-ghost">
              时延分析
            </Link>
          </div>
        </div>
        <aside className="hero-aside">
          <div className="metric-grid">
            <div className="metric-card">
              <span className="stat-label">工作目录</span>
              <p className="path-text is-compact" title={overview?.runs_dir ?? "--"}>
                {overview?.runs_dir ?? "--"}
              </p>
            </div>
            <div className="metric-card">
              <span className="stat-label">项目数</span>
              <strong>{overview?.project_count ?? 0}</strong>
            </div>
            <div className="metric-card">
              <span className="stat-label">运行总数</span>
              <strong>{overview?.run_count ?? 0}</strong>
            </div>
          </div>
          {latestRun ? (
            <div className="section-panel compact-panel overview-highlight">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">最近一次运行</p>
                  <h2>{latestRun.run_name}</h2>
                </div>
                <Link href={runHref(latestRun)} className="text-link">
                  查看详情 →
                </Link>
              </div>
              <div className="overview-list">
                <div className="overview-row">
                  <span className="stat-label">项目</span>
                  <strong>{latestRun.project}</strong>
                </div>
                <div className="overview-row">
                  <span className="stat-label">模型</span>
                  <span title={latestRun.model_name || "unknown"}>
                    {formatShortText(latestRun.model_name || "unknown", 48)}
                  </span>
                </div>
                <div className="overview-row">
                  <span className="stat-label">任务</span>
                  <span title={latestRun.task_name || "--"}>
                    {formatShortText(latestRun.task_name || "--", 72)}
                  </span>
                </div>
                <div className="overview-row">
                  <span className="stat-label">更新时间</span>
                  <span>{formatDate(latestRun.updated_at)}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-panel">当前目录暂无运行记录</div>
          )}
        </aside>
      </section>

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">快捷入口</p>
            <h2>核心功能</h2>
          </div>
        </div>
        <div className="feature-grid">
          <Link href="/runs" className="feature-card feature-card-link">
            <h3>推理日志</h3>
            <p>浏览项目和 run 列表，逐帧回放、轨迹分析与 attention 可视化。</p>
          </Link>
          <Link href="/latency" className="feature-card feature-card-link">
            <h3>时延对比</h3>
            <p>多 run transport / inference / total 延迟对比与分布分析。</p>
          </Link>
          <Link href="/datasets" className="feature-card feature-card-link">
            <h3>数据集浏览</h3>
            <p>查看 episode 图像网格、动作序列和工作空间点云。</p>
          </Link>
          <Link href="/eval" className="feature-card feature-card-link">
            <h3>开环评估</h3>
            <p>读取目录或 JSON 结果，检查预测轨迹、误差分布和静态可视化。</p>
          </Link>
        </div>
      </section>

      {latestRuns.length > 0 ? (
        <section className="section-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">最近运行</p>
              <h2>最新 {latestRuns.length} 条记录</h2>
            </div>
            <Link href="/runs" className="text-link">
              查看全部 →
            </Link>
          </div>
          <div className="run-card-grid">
            {latestRuns.map((run) => (
              <RunCard key={run.run_id} run={run} />
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
