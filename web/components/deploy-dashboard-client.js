"use client";

import { useEffect, useRef, useState, useTransition } from "react";

import { browserFetchJson, runDeployCommand, saveDeployInputs, stopDeployJob } from "@/lib/api";

const DEPLOY_FORM_STORAGE_KEY = "vlalab.deploy.form.v1";
const DEPLOY_INPUT_HISTORY_KEY = "vlalab.deploy.input-history.v1";
const MAX_HISTORY_PER_FIELD = 8;

const EMPTY_OVERVIEW = {
  refreshed_at: "",
  targets: [],
  inputs: [],
  commands: [],
  jobs: [],
};

function prettyState(state) {
  if (!state) return "Unknown";
  return String(state)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function prettyKey(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function stateToneClass(state) {
  if (!state) return "is-neutral";
  if (["running", "queued"].includes(state)) return "is-running";
  if (["success", "connected", "ready"].includes(state)) return "is-ready";
  if (["failed", "disconnected"].includes(state)) return "is-offline";
  return "is-neutral";
}

function buildInitialForm(overview) {
  const next = {};
  for (const input of overview?.inputs || []) {
    next[input.id] = input.current_value ?? input.default ?? "";
  }
  return next;
}

function mergeFormWithInputs(current, inputs) {
  const next = { ...current };
  for (const input of inputs || []) {
    if (!(input.id in next) || next[input.id] === undefined || next[input.id] === null) {
      next[input.id] = input.default || "";
    }
  }
  return next;
}

function pruneValues(values) {
  return Object.fromEntries(
    Object.entries(values || {}).filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== "")
  );
}

function loadStoredObject(key) {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (error) {
    return {};
  }
}

function rememberValue(history, fieldId, value) {
  const normalized = String(value || "").trim();
  if (!normalized) {
    return history;
  }
  const current = Array.isArray(history?.[fieldId]) ? history[fieldId] : [];
  return {
    ...history,
    [fieldId]: [normalized, ...current.filter((item) => item !== normalized)].slice(0, MAX_HISTORY_PER_FIELD),
  };
}

function formatTime(value) {
  if (!value) return "--";
  try {
    return new Date(value).toLocaleString();
  } catch (error) {
    return value;
  }
}

function ConnectionCard({ target }) {
  return (
    <article className="deploy-target-card">
      <div className="deploy-target-header">
        <div>
          <p className="eyebrow">{target.id}</p>
          <h3>{target.label}</h3>
        </div>
        <span className={`deploy-status-chip ${stateToneClass(target.connected ? "connected" : "disconnected")}`}>
          {target.connected ? "Connected" : "Disconnected"}
        </span>
      </div>
      <div className="deploy-meta-list">
        <div className="deploy-meta-row">
          <span className="stat-label">Target</span>
          <span>{target.id}</span>
        </div>
        <div className="deploy-meta-row">
          <span className="stat-label">Status</span>
          <span>{target.connected ? "SSH reachable" : "SSH unavailable"}</span>
        </div>
        {target.last_error ? (
          <div className="deploy-meta-row is-error">
            <span className="stat-label">Error</span>
            <span>{target.last_error}</span>
          </div>
        ) : null}
      </div>
    </article>
  );
}

function InputField({ spec, value, history, onChange, onCommit }) {
  if (spec.type === "enum") {
    return (
      <label>
        <span className="stat-label">{spec.label}</span>
        <select value={value || ""} onChange={(event) => onChange(spec.id, event.target.value)}>
          {(spec.options || []).map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
    );
  }

  const datalistId = `deploy-history-${spec.id}`;

  return (
    <label className="deploy-field-span">
      <span className="stat-label">{spec.label}</span>
      <input
        type="text"
        list={history?.length ? datalistId : undefined}
        value={value || ""}
        onChange={(event) => onChange(spec.id, event.target.value)}
        onBlur={(event) => onCommit(spec.id, event.target.value)}
        placeholder={spec.default || (spec.type === "path" ? "/remote/path/to/config" : "")}
      />
      {history?.length ? (
        <datalist id={datalistId}>
          {history.map((item) => (
            <option key={item} value={item} />
          ))}
        </datalist>
      ) : null}
    </label>
  );
}

function CommandCard({ command, target, missingInputs, busy, activeJob, stopping, onRun, onStop }) {
  const isDisconnected = !target?.connected;
  const isRunning = busy;
  let state = "ready";
  let summary = "Ready to run";

  if (isRunning) {
    state = activeJob?.state || "running";
    summary =
      activeJob?.state === "stopping"
        ? "Stopping current job"
        : activeJob?.id
          ? `Job ${activeJob.id} is in progress`
          : "A job for this command is already running";
  } else if (isDisconnected) {
    state = "disconnected";
    summary = target?.last_error || "SSH target is not reachable";
  } else if (missingInputs.length) {
    state = "queued";
    summary = `Missing inputs: ${missingInputs.map((item) => prettyKey(item)).join(", ")}`;
  }

  return (
    <article className="deploy-runbook-card">
      <div className="deploy-runbook-head">
        <div>
          <p className="eyebrow">{target?.label || command.target_id}</p>
          <h3>{command.label}</h3>
        </div>
        <span className={`deploy-status-chip ${stateToneClass(state)}`}>{prettyState(state)}</span>
      </div>
      <pre className="deploy-command-block">{command.resolved_preview || "No preview available"}</pre>
      <div className="deploy-command-meta">
        <div>
          <span className="stat-label">Mode</span>
          <code>{command.background ? "background" : "foreground"}</code>
        </div>
        <div>
          <span className="stat-label">Inputs</span>
          <code>{command.required_inputs?.length ? command.required_inputs.join(", ") : "--"}</code>
        </div>
        <div>
          <span className="stat-label">Target</span>
          <code>{command.target_id}</code>
        </div>
      </div>
      <p className="muted">{summary}</p>
      {missingInputs.length ? (
        <div className="deploy-manual-hint">缺少参数: {missingInputs.map((item) => prettyKey(item)).join(", ")}</div>
      ) : null}
      <div className="deploy-action-row">
        <button
          type="button"
          className="deploy-action-button is-primary"
          onClick={() => onRun(command.id)}
          disabled={isDisconnected || missingInputs.length > 0 || isRunning}
        >
          运行命令
        </button>
        {activeJob?.stoppable ? (
          <button
            type="button"
            className="deploy-action-button is-danger"
            onClick={() => onStop(activeJob.id)}
            disabled={stopping}
          >
            {stopping ? "停止中..." : "停止命令"}
          </button>
        ) : null}
      </div>
    </article>
  );
}

function JobCard({ command, targetLabel, job, stopping, onStop }) {
  const state = job?.state || "idle";
  const stdoutPath = job?.stdout_log || (command.background ? "等待命令首次运行后生成" : "--");
  const stderrPath = job?.stderr_log || (command.background ? "等待命令首次运行后生成" : "--");
  const hasLogs = Boolean(job?.last_stdout || job?.last_stderr);

  return (
    <article className="deploy-workflow-card">
      <div className="deploy-workflow-top">
        <div className="deploy-target-header">
          <div>
            <p className="eyebrow">{targetLabel}</p>
            <h3>{command.label}</h3>
          </div>
          <span className={`deploy-status-chip ${stateToneClass(state)}`}>{prettyState(state)}</span>
        </div>
        <div className="deploy-meta-list">
          <div className="deploy-meta-row">
            <span className="stat-label">Job ID</span>
            <code>{job?.id || "--"}</code>
          </div>
          <div className="deploy-meta-row">
            <span className="stat-label">Remote PID</span>
            <span>{job?.remote_pid || "--"}</span>
          </div>
          <div className="deploy-meta-row">
            <span className="stat-label">Submitted</span>
            <span>{formatTime(job?.submitted_at)}</span>
          </div>
          <div className="deploy-meta-row">
            <span className="stat-label">Started</span>
            <span>{formatTime(job?.started_at)}</span>
          </div>
          <div className="deploy-meta-row">
            <span className="stat-label">Finished</span>
            <span>{formatTime(job?.finished_at)}</span>
          </div>
          {job?.error ? (
            <div className="deploy-meta-row is-error">
              <span className="stat-label">Error</span>
              <span>{job.error}</span>
            </div>
          ) : null}
        </div>

        <div className="deploy-command-meta">
          <div>
            <span className="stat-label">stdout</span>
            <code>{stdoutPath}</code>
          </div>
          <div>
            <span className="stat-label">stderr</span>
            <code>{stderrPath}</code>
          </div>
        </div>

        <div className="deploy-action-row">
          {job?.stoppable ? (
            <button
              type="button"
              className="deploy-action-button is-danger"
              onClick={() => onStop(job.id)}
              disabled={stopping}
            >
              {stopping ? "停止中..." : "停止命令"}
            </button>
          ) : (
            <span className="muted">{job ? "当前任务不可停止" : "该命令最近还没有执行记录"}</span>
          )}
        </div>
      </div>

      <div className="deploy-log-stack">
        <div className="deploy-log-panel is-stdout">
          <div className="deploy-log-header">
            <span className="stat-label">stdout · 最近一次执行</span>
          </div>
          {job?.last_stdout ? (
            <pre className="deploy-command-block is-output is-scrollable">{job.last_stdout}</pre>
          ) : (
            <div className="empty-panel deploy-empty-panel deploy-log-empty">
              {hasLogs ? "stdout 暂无输出" : "还没有这条命令的 stdout 记录"}
            </div>
          )}
        </div>
        <div className="deploy-log-panel is-stderr">
          <div className="deploy-log-header">
            <span className="stat-label">stderr · 最近一次执行</span>
          </div>
          {job?.last_stderr ? (
            <pre className="deploy-command-block is-output is-error is-scrollable">{job.last_stderr}</pre>
          ) : (
            <div className="empty-panel deploy-empty-panel deploy-log-empty">
              {hasLogs ? "stderr 暂无输出" : "还没有这条命令的 stderr 记录"}
            </div>
          )}
        </div>
      </div>
    </article>
  );
}

export default function DeployDashboardClient({ initialOverview }) {
  const seededOverview = initialOverview || EMPTY_OVERVIEW;
  const importInputRef = useRef(null);
  const [overview, setOverview] = useState(seededOverview);
  const [jobs, setJobs] = useState(seededOverview.jobs || []);
  const [form, setForm] = useState(() =>
    mergeFormWithInputs(
      {
        ...buildInitialForm(seededOverview),
        ...loadStoredObject(DEPLOY_FORM_STORAGE_KEY),
      },
      seededOverview.inputs
    )
  );
  const [inputHistory, setInputHistory] = useState(() => loadStoredObject(DEPLOY_INPUT_HISTORY_KEY));
  const [errorText, setErrorText] = useState("");
  const [flashMessage, setFlashMessage] = useState("");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isSavingInputs, setIsSavingInputs] = useState(false);
  const [activeCommandId, setActiveCommandId] = useState("");
  const [stoppingJobId, setStoppingJobId] = useState("");
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    setForm((current) => mergeFormWithInputs(current, overview.inputs));
  }, [overview.inputs]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(DEPLOY_FORM_STORAGE_KEY, JSON.stringify(form));
  }, [form]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(DEPLOY_INPUT_HISTORY_KEY, JSON.stringify(inputHistory));
  }, [inputHistory]);

  async function fetchDeployState(values) {
    const [overviewResult, jobsResult] = await Promise.allSettled([
      browserFetchJson("/api/deploy/overview", values),
      browserFetchJson("/api/deploy/jobs"),
    ]);

    if (overviewResult.status !== "fulfilled") {
      throw overviewResult.reason;
    }

    const nextOverview = overviewResult.value;
    const nextJobs =
      jobsResult.status === "fulfilled"
        ? jobsResult.value?.jobs || []
        : nextOverview?.jobs || [];

    return {
      overview: nextOverview,
      jobs: nextJobs,
      jobsError: jobsResult.status === "rejected" ? jobsResult.reason : null,
    };
  }

  useEffect(() => {
    let active = true;

    async function poll() {
      if (!active) {
        return;
      }
      try {
        const values = pruneValues(form);
        const snapshot = await fetchDeployState(values);
        if (!active) {
          return;
        }
        setOverview(snapshot.overview);
        setJobs(snapshot.jobs);
        setErrorText(snapshot.jobsError?.message || "");
      } catch (error) {
        if (!active) {
          return;
        }
        setErrorText(error.message || "无法读取 deploy 状态，请确认 API 已启动且 deploy 配置有效。");
      }
    }

    poll();
    const timer = window.setInterval(poll, 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [form.model_server_config_path, form.inference_client_config_path]);

  async function refreshNow() {
    setIsRefreshing(true);
    setErrorText("");
    try {
      const values = pruneValues(form);
      const snapshot = await fetchDeployState(values);
      setOverview(snapshot.overview);
      setJobs(snapshot.jobs);
      setErrorText(snapshot.jobsError?.message || "");
    } catch (error) {
      setErrorText(error.message || "无法读取 deploy 状态，请确认 vlalab API 已启动。");
    } finally {
      setIsRefreshing(false);
    }
  }

  function updateField(key, value) {
    setForm((current) => ({
      ...current,
      [key]: value,
    }));
  }

  async function persistInputs(nextValues, successMessage = "") {
    setIsSavingInputs(true);
    try {
      const response = await saveDeployInputs(nextValues);
      if (successMessage) {
        setFlashMessage(successMessage);
      }
      return response?.values || nextValues;
    } catch (error) {
      setErrorText(error.message || "输入参数保存失败");
      return nextValues;
    } finally {
      setIsSavingInputs(false);
    }
  }

  function commitFieldValue(key, value) {
    setInputHistory((current) => rememberValue(current, key, value));
    const nextValues = {
      ...form,
      [key]: value,
    };
    void persistInputs(nextValues);
  }

  function handleRun(commandId) {
    setErrorText("");
    setFlashMessage("");
    setActiveCommandId(commandId);
    const currentValues = pruneValues(form);

    for (const [key, value] of Object.entries(currentValues)) {
      commitFieldValue(key, value);
    }

    startTransition(async () => {
      try {
        const response = await runDeployCommand(commandId, currentValues);
        setFlashMessage(response.message || "命令已提交");
        if (response.job) {
          setJobs((current) => [response.job, ...current.filter((item) => item.id !== response.job.id)]);
        }
        const snapshot = await fetchDeployState(currentValues);
        setOverview(snapshot.overview);
        setJobs(snapshot.jobs);
        setErrorText(snapshot.jobsError?.message || "");
      } catch (error) {
        setErrorText(error.message || "命令执行失败");
      } finally {
        setActiveCommandId("");
      }
    });
  }

  async function handleStop(jobId) {
    setErrorText("");
    setFlashMessage("");
    setStoppingJobId(jobId);
    try {
      const response = await stopDeployJob(jobId);
      setFlashMessage(response.message || "停止请求已发送");
      if (response.job) {
        setJobs((current) => [response.job, ...current.filter((item) => item.id !== response.job.id)]);
      }
      await refreshNow();
    } catch (error) {
      setErrorText(error.message || "停止命令失败");
    } finally {
      setStoppingJobId("");
    }
  }

  function handleExportJson() {
    const payload = JSON.stringify(pruneValues(form), null, 2);
    const blob = new Blob([payload], { type: "application/json" });
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "vlalab-deploy-inputs.json";
    anchor.click();
    window.URL.revokeObjectURL(url);
  }

  function handleOpenImport() {
    importInputRef.current?.click();
  }

  async function handleImportJson(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      if (!parsed || typeof parsed !== "object") {
        throw new Error("JSON 内容必须是对象");
      }

      const allowedKeys = new Set((overview.inputs || []).map((item) => item.id));
      const nextValues = { ...form };
      const nextHistory = { ...inputHistory };

      for (const [key, value] of Object.entries(parsed)) {
        if (!allowedKeys.has(key)) {
          continue;
        }
        const normalized = String(value ?? "").trim();
        nextValues[key] = normalized;
        nextHistory[key] = rememberValue(nextHistory, key, normalized)[key] || nextHistory[key] || [];
      }

      setForm(nextValues);
      setInputHistory(nextHistory);
      await persistInputs(nextValues, "JSON 配置已导入");
      await refreshNow();
    } catch (error) {
      setErrorText(error.message || "JSON 导入失败");
    }
  }

  const targets = overview.targets || [];
  const commands = overview.commands || [];
  const inputs = overview.inputs || [];
  const targetMap = Object.fromEntries(targets.map((target) => [target.id, target]));
  const latestJobByCommand = Object.fromEntries(
    commands.map((command) => [
      command.id,
      jobs.find((job) => job.command_id === command.id) || null,
    ])
  );
  const connectedCount = targets.filter((target) => target.connected).length;
  const runningJobs = jobs.filter((job) => ["queued", "running", "stopping"].includes(job.state)).length;
  const successfulJobs = jobs.filter((job) => job.state === "success").length;

  return (
    <div className="section-stack">
      <section className="hero-panel deploy-hero">
        <div className="hero-copy">
          <p className="eyebrow">Deploy Console</p>
          <h1>RealWorld Deploy Dashboard</h1>
          <p className="hero-desc">
            这个版本只负责 2 台机器的 SSH 连接、4 条固定命令的后台执行，以及日志与 PID 状态展示。
          </p>
          <div className="hero-actions">
            <button type="button" className="button-link deploy-inline-button is-secondary" onClick={refreshNow} disabled={isRefreshing}>
              {isRefreshing ? "刷新中..." : "立即刷新"}
            </button>
            <span className="deploy-refresh-note">
              最近刷新 {overview?.refreshed_at ? new Date(overview.refreshed_at).toLocaleTimeString() : "--"}
            </span>
          </div>
        </div>
        <aside className="hero-aside">
          <div className="metric-grid">
            <div className="metric-card">
              <span className="stat-label">Connected Targets</span>
              <strong>{connectedCount}</strong>
            </div>
            <div className="metric-card">
              <span className="stat-label">Commands</span>
              <strong>{commands.length}</strong>
            </div>
            <div className="metric-card">
              <span className="stat-label">Running Jobs</span>
              <strong>{runningJobs}</strong>
            </div>
            <div className="metric-card">
              <span className="stat-label">Successful Jobs</span>
              <strong>{successfulJobs}</strong>
            </div>
          </div>
          {flashMessage ? <div className="deploy-banner is-success">{flashMessage}</div> : null}
          {errorText ? <div className="deploy-banner is-error">{errorText}</div> : null}
        </aside>
      </section>

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Connections</p>
            <h2>两台目标机器自动连接检查</h2>
          </div>
        </div>
        <div className="deploy-target-grid">
          {targets.map((target) => (
            <ConnectionCard key={target.id} target={target} />
          ))}
        </div>
      </section>

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Inputs</p>
            <h2>固定输入参数</h2>
          </div>
          <div className="deploy-action-row">
            <button
              type="button"
              className="deploy-action-button is-ghost"
              onClick={() => persistInputs(form, "输入参数已保存")}
              disabled={isSavingInputs}
            >
              {isSavingInputs ? "保存中..." : "保存记忆"}
            </button>
            <button type="button" className="deploy-action-button is-ghost" onClick={handleExportJson}>
              导出 JSON
            </button>
            <button type="button" className="deploy-action-button is-ghost" onClick={handleOpenImport}>
              导入 JSON
            </button>
            <input ref={importInputRef} type="file" accept="application/json,.json" hidden onChange={handleImportJson} />
          </div>
        </div>
        <div className="deploy-form-grid">
          {inputs.map((input) => (
            <InputField
              key={input.id}
              spec={input}
              value={form[input.id] || ""}
              history={inputHistory[input.id] || []}
              onChange={updateField}
              onCommit={commitFieldValue}
            />
          ))}
        </div>
      </section>

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Commands</p>
            <h2>固定四个部署命令</h2>
          </div>
        </div>
        <div className="deploy-runbook-grid">
          {commands.map((command) => {
            const missingInputs = (command.required_inputs || []).filter((key) => !String(form[key] || "").trim());
            const activeJob = jobs.find(
              (job) => job.command_id === command.id && ["queued", "running", "stopping"].includes(job.state)
            );
            return (
              <CommandCard
                key={command.id}
                command={command}
                target={targetMap[command.target_id]}
                missingInputs={missingInputs}
                busy={Boolean(activeJob) || (isPending && activeCommandId === command.id)}
                activeJob={activeJob}
                stopping={stoppingJobId === activeJob?.id}
                onRun={handleRun}
                onStop={handleStop}
              />
            );
          })}
        </div>
      </section>

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Jobs</p>
            <h2>最近执行记录与日志（和部署命令一一对应）</h2>
          </div>
        </div>
        <div className="deploy-workflow-grid">
          {commands.length ? (
            commands.map((command) => (
              <JobCard
                key={command.id}
                command={command}
                job={latestJobByCommand[command.id]}
                targetLabel={targetMap[command.target_id]?.label || command.target_id}
                stopping={stoppingJobId === latestJobByCommand[command.id]?.id}
                onStop={handleStop}
              />
            ))
          ) : (
            <div className="empty-panel deploy-empty-panel">当前没有可展示的部署命令。</div>
          )}
        </div>
      </section>
    </div>
  );
}
