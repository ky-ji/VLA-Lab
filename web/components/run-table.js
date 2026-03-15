import Link from "next/link";

import { formatDate, formatMs, formatShortText } from "@/lib/format";

function runHref(run) {
  return `/runs/${encodeURIComponent(run.project)}/${encodeURIComponent(run.run_name)}`;
}

export default function RunTable({ runs, showProject = true }) {
  if (!runs || runs.length === 0) {
    return <div className="empty-panel">暂无可显示的运行记录</div>;
  }

  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            <th>运行名称</th>
            {showProject ? <th>项目</th> : null}
            <th>模型</th>
            <th>任务</th>
            <th>步数</th>
            <th>平均延迟</th>
            <th>更新时间</th>
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
              <td>{formatShortText(run.task_name || "--", 32)}</td>
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
