"use client";

import { useEffect, useState } from "react";

import RunReplayClient from "@/components/run-replay-client";
import { browserFetchJson } from "@/lib/api";

export default function RunDetailPageClient({ project, runName }) {
  const [replay, setReplay] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadReplay() {
      setLoading(true);
      setError("");

      try {
        const payload = await browserFetchJson(
          `/api/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/replay`
        );
        if (!cancelled) {
          setReplay(payload);
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
        <p>正在加载 run 详情 ...</p>
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
