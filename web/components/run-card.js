import Link from "next/link";

import { formatDate, formatShortText } from "@/lib/format";

function runHref(run) {
  return `/runs/${encodeURIComponent(run.project)}/${encodeURIComponent(run.run_name)}`;
}

export default function RunCard({ run }) {
  return (
    <article className="run-card">
      <div className="run-card-header">
        <span className="badge">{run.project}</span>
        <span className="muted">{formatDate(run.updated_at)}</span>
      </div>
      <h3>{run.run_name}</h3>
      <p>{formatShortText(run.task_name || "No task name")}</p>
      <dl className="inline-meta">
        <div>
          <dt>Model</dt>
          <dd>{formatShortText(run.model_name || "unknown", 24)}</dd>
        </div>
        <div>
          <dt>Steps</dt>
          <dd>{run.total_steps ?? 0}</dd>
        </div>
      </dl>
      <Link href={runHref(run)} className="text-link">
        Open run
      </Link>
    </article>
  );
}
