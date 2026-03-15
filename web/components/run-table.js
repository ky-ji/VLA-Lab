import Link from "next/link";

import { formatDate, formatMs, formatShortText } from "@/lib/format";

function runHref(run) {
  return `/runs/${encodeURIComponent(run.project)}/${encodeURIComponent(run.run_name)}`;
}

export default function RunTable({ runs, showProject = true }) {
  if (!runs || runs.length === 0) {
    return <div className="empty-panel">还没有可显示的 run。</div>;
  }

  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            <th>Run</th>
            {showProject ? <th>Project</th> : null}
            <th>Model</th>
            <th>Task</th>
            <th>Steps</th>
            <th>Total Avg</th>
            <th>Updated</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id}>
              <td>
                <Link href={runHref(run)} className="table-link">
                  {run.run_name}
                </Link>
              </td>
              {showProject ? <td>{run.project}</td> : null}
              <td>{formatShortText(run.model_name || "unknown", 26)}</td>
              <td>{formatShortText(run.task_name || "unknown", 32)}</td>
              <td>{run.total_steps ?? 0}</td>
              <td>{formatMs(run.latency?.total_latency?.avg_ms)}</td>
              <td>{formatDate(run.updated_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
