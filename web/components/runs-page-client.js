"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import RunTable from "@/components/run-table";
import { browserFetchJson } from "@/lib/api";

const INITIAL_STATE = {
  runs_dir: null,
  total: 0,
  items: [],
};

export default function RunsPageClient() {
  const [response, setResponse] = useState(INITIAL_STATE);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadRuns() {
      setLoading(true);
      setError("");

      try {
        const payload = await browserFetchJson("/api/runs", {
          limit: 200,
        });
        if (!cancelled) {
          setResponse(payload || INITIAL_STATE);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : "加载 run 列表失败");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadRuns();
    return () => {
      cancelled = true;
    };
  }, []);

  const runs = response?.items || [];
  const projects = useMemo(
    () => Array.from(new Set(runs.map((run) => run.project))).sort(),
    [runs]
  );

  return (
    <div className="section-stack">
      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Runs</p>
            <h2>运行记录</h2>
          </div>
          <Link href="/latency" className="text-link">
            去做时延对比 →
          </Link>
        </div>
        <div className="metric-grid">
          <div className="metric-card">
            <span className="stat-label">工作目录</span>
            <p className="path-text is-compact" title={response?.runs_dir ?? "--"}>
              {response?.runs_dir ?? "--"}
            </p>
          </div>
          <div className="metric-card">
            <span className="stat-label">项目数</span>
            <strong>{projects.length}</strong>
          </div>
          <div className="metric-card">
            <span className="stat-label">运行总数</span>
            <strong>{response?.total ?? runs.length}</strong>
          </div>
        </div>
      </section>

      <section className="section-panel">
        {loading ? (
          <div className="placeholder-panel">
            <p>正在加载 run 列表 ...</p>
          </div>
        ) : null}
        {error ? (
          <div className="empty-panel">
            <p>加载失败: {error}</p>
          </div>
        ) : null}
        {!loading && !error ? <RunTable runs={runs} showProject /> : null}
      </section>
    </div>
  );
}
