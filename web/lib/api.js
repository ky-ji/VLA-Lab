const DEFAULT_API_BASE =
  process.env.VLALAB_API_BASE_URL ||
  process.env.NEXT_PUBLIC_VLALAB_API_BASE_URL ||
  "http://127.0.0.1:8000";

export function getApiBase() {
  return DEFAULT_API_BASE.replace(/\/$/, "");
}

export function toPublicApiUrl(path) {
  if (!path) {
    return "";
  }
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${getApiBase()}${path}`;
}

function buildUrl(path, params = {}) {
  const url = new URL(path, `${getApiBase()}/`);
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        if (item !== undefined && item !== null && item !== "") {
          url.searchParams.append(key, item);
        }
      }
      continue;
    }
    url.searchParams.set(key, value);
  }
  return url.toString();
}

async function fetchJson(path, params = {}, options = {}) {
  const { allow404 = false } = options;
  const response = await fetch(buildUrl(path, params), {
    cache: "no-store",
  });

  if (allow404 && response.status === 404) {
    return null;
  }

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

async function safeFetch(path, params = {}, options = {}) {
  try {
    return await fetchJson(path, params, options);
  } catch (error) {
    return null;
  }
}

export async function browserFetchJson(path, params = {}, options = {}) {
  return fetchJson(path, params, options);
}

export async function browserPostJson(path, body = {}, options = {}) {
  const response = await fetch(buildUrl(path), {
    method: options.method || "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = payload?.detail || payload?.message || JSON.stringify(payload);
    } catch (error) {
      detail = await response.text();
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

export async function browserDelete(path) {
  const response = await fetch(buildUrl(path), {
    method: "DELETE",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

export async function getOverview() {
  return safeFetch("/api/overview");
}

export async function getProjects() {
  return safeFetch("/api/projects");
}

export async function getRuns(params = {}) {
  return safeFetch("/api/runs", params);
}

export async function getRunDetail(project, runName) {
  return safeFetch(`/api/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}`, {}, { allow404: true });
}

export async function getRunReplay(project, runName) {
  return safeFetch(`/api/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/replay`, {}, { allow404: true });
}

export async function getRunSteps(project, runName, params = {}) {
  return safeFetch(
    `/api/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/steps`,
    params,
    { allow404: true }
  );
}

export async function getLatencyCompare(runs) {
  return safeFetch("/api/latency/compare", { runs });
}

export async function getDeployOverview(params = {}) {
  return safeFetch("/api/deploy/overview", params);
}

export async function runDeployCommand(commandId, values = {}) {
  return browserPostJson("/api/deploy/run", {
    command_id: commandId,
    values,
  });
}

export async function getDeployJobs() {
  return safeFetch("/api/deploy/jobs");
}

export async function getDeployJobLogs(jobId, params = {}) {
  return safeFetch(`/api/deploy/jobs/${encodeURIComponent(jobId)}/logs`, params);
}

export async function getRunAttention(project, runName, params = {}) {
  return safeFetch(`/api/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/attention`, params, { allow404: true });
}

export async function getDatasetInspect(path) {
  return safeFetch("/api/datasets/inspect", { path });
}

export async function getDatasetEpisode(params = {}) {
  return safeFetch("/api/datasets/episode", params);
}

export async function getEvalView(params = {}) {
  return safeFetch("/api/eval/view", params);
}
