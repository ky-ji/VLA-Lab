import Link from "next/link";

import { formatDate, formatShortText } from "@/lib/format";

function runHref(run) {
  return `/runs/${encodeURIComponent(run.project)}/${encodeURIComponent(run.run_name)}`;
}

export default function RunCard({ run }) {
  return (
    <Link href={runHref(run)} className="run-card run-card-link">
      <div className="run-card-header">
        <span className="badge">{run.project}</span>
        <span className="muted">{formatDate(run.updated_at)}</span>
      </div>
      <h3>{run.run_name}</h3>
      <p className="run-card-task">{formatShortText(run.task_name || "未设置任务名", 56)}</p>
      <dl className="inline-meta">
        <div>
          <dt>模型</dt>
          <dd>{formatShortText(run.model_name || "unknown", 24)}</dd>
        </div>
        <div>
          <dt>步数</dt>
          <dd>{run.total_steps ?? 0}</dd>
        </div>
      </dl>
    </Link>
  );
}
