import { getProjects, getRuns } from "@/lib/api";
import RunTable from "@/components/run-table";

export const dynamic = "force-dynamic";

export default async function RunsPage({ searchParams }) {
  const params = (await searchParams) ?? {};
  const selectedProject = typeof params.project === "string" ? params.project : "";
  const query = typeof params.q === "string" ? params.q : "";

  const [projectsResponse, runsResponse] = await Promise.all([
    getProjects(),
    getRuns({
      project: selectedProject || undefined,
      search: query || undefined,
      limit: 200,
      include_latency: true,
    }),
  ]);

  const projects = projectsResponse?.projects ?? [];
  const runs = runsResponse?.items ?? [];

  return (
    <div className="section-stack">
      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">推理日志</p>
            <h2>运行记录浏览</h2>
          </div>
        </div>
        <form action="/runs" className="form-grid">
          <input
            type="search"
            name="q"
            defaultValue={query}
            placeholder="搜索 run / project / model / task ..."
          />
          <select name="project" defaultValue={selectedProject}>
            <option value="">全部项目</option>
            {projects.map((project) => (
              <option key={project} value={project}>
                {project}
              </option>
            ))}
          </select>
          <button type="submit">筛选</button>
        </form>
      </section>

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">结果</p>
            <h2>共 {runs.length} 条记录</h2>
          </div>
        </div>
        <RunTable runs={runs} showProject={!selectedProject} />
      </section>
    </div>
  );
}
