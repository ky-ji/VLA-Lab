from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_component(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_replay_workspace_has_lightweight_signal_panels():
    viewer = read_component("web/components/run-workspace-viewer.js")
    replay = read_component("web/components/run-replay-client.js")

    assert "signalPanels" in viewer
    assert "run-workspace-signal-panel" in viewer
    assert "replaySignalPanels" in replay
    assert "State XYZ" in replay
    assert "Action XYZ" in replay


def test_replay_trajectory_is_not_rendered_during_playback():
    viewer = read_component("web/components/run-workspace-viewer.js")
    replay = read_component("web/components/run-replay-client.js")

    assert "showTrajectory" in viewer
    assert "showTrajectory={false}" in replay


def test_playback_uses_animation_frame_ticker():
    replay = read_component("web/components/run-replay-client.js")

    assert "requestAnimationFrame" in replay
    assert "lastPlaybackTickRef" in replay


def test_replay_workspace_removes_fixed_bottom_dock():
    replay = read_component("web/components/run-replay-client.js")
    css = read_component("web/app/globals.css")

    assert "<RunWorkspaceTimeline" not in replay
    assert "run-workspace-timeline-dock" not in css


def test_replay_workspace_keeps_inline_playback_and_images_together():
    viewer = read_component("web/components/run-workspace-viewer.js")
    css = read_component("web/app/globals.css")

    assert "run-workspace-inline-playback" in viewer
    assert "run-workspace-visual-stack" in viewer
    assert ".run-workspace-visual-stack" in css


def test_replay_window_loading_uses_request_pool_not_global_blocker():
    replay = read_component("web/components/run-replay-client.js")
    window_section = replay[replay.index("const requestStepWindow") : replay.index("const isStepFrameReady")]

    assert "stepWindowLoading" not in replay
    assert "pendingWindowKeys" in replay
    assert "requestedWindowKeysRef" in replay
    assert "requestStepWindow" in replay
    assert "if (!ignore)" not in window_section


def test_replay_viewer_has_exact_frame_loading_and_data_dock():
    viewer = read_component("web/components/run-workspace-viewer.js")
    replay = read_component("web/components/run-replay-client.js")

    assert "frameStatus" in viewer
    assert "Loading exact frame" in viewer
    assert "No images recorded for step" in viewer
    assert "RunWorkspaceStepDataDock" in viewer
    assert "run-workspace-data-dock" in viewer
    assert "FrameContextBand" not in viewer
    assert "frameStatus={frameStatus}" in replay


def test_dataset_viewer_exposes_candidates_and_lerobot_curves():
    dataset = read_component("web/components/dataset-viewer-client.js")

    assert "/api/datasets/candidates" in dataset
    assert "/data3/jikangye/vla-data/008_stack_bowls" in dataset
    assert "dataset-candidate-card" in dataset
    assert "State XYZ series" in dataset
    assert "current_state" in dataset


def test_dataset_viewer_uses_runs_style_workspace_layout():
    dataset = read_component("web/components/dataset-viewer-client.js")
    css = read_component("web/app/globals.css")

    assert "dataset-workspace-shell" in dataset
    assert "dataset-workspace-header" in dataset
    assert "dataset-workspace-rail" in dataset
    assert "dataset-workspace-viewer" in dataset
    assert "dataset-workspace-inspector" in dataset
    assert "dataset-workspace-mode-button" in dataset
    assert "dataset-workspace-candidate-card" in dataset
    assert "Dataset Overview" in dataset
    assert "Episode Replay" in dataset
    assert "Workspace" in dataset
    assert ".dataset-workspace-grid" in css
    assert "grid-template-columns: 240px minmax(0, 1fr) 300px;" in css


def test_dataset_viewer_keeps_step_images_curves_and_current_values_visible():
    dataset = read_component("web/components/dataset-viewer-client.js")

    assert "dataset-workspace-stepbar" in dataset
    assert "dataset-workspace-image-grid" in dataset
    assert "dataset-workspace-chart-grid" in dataset
    assert "Current Action" in dataset
    assert "Current State" in dataset
    assert "XYZ action series" in dataset
    assert "State XYZ series" in dataset
    assert "Gripper series" in dataset
    assert "SimpleTabs" not in dataset


def test_dataset_viewer_supports_episode_playback_controls():
    dataset = read_component("web/components/dataset-viewer-client.js")
    css = read_component("web/app/globals.css")

    assert "isDatasetPlaying" in dataset
    assert "datasetPlaybackSpeed" in dataset
    assert "datasetPlaybackTimerRef" in dataset
    assert "setInterval" in dataset
    assert "clearInterval" in dataset
    assert "episodeLoading" in dataset
    assert "dataset-workspace-playback" in dataset
    assert "dataset-workspace-play-button" in dataset
    assert "Dataset playback speed" in dataset
    assert "Play" in dataset
    assert "Pause" in dataset
    assert ".dataset-workspace-playback" in css
    assert ".dataset-workspace-play-button" in css


def test_eval_viewer_exposes_default_openloop_candidates():
    eval_viewer = read_component("web/components/eval-viewer-client.js")
    css = read_component("web/app/globals.css")

    assert "/api/eval/candidates" in eval_viewer
    assert "/data3/jikangye/workspace/baselines/vla-baselines/Isaac-GR00T/outputs/vlalab_eval" in eval_viewer
    assert "eval-candidate-card" in eval_viewer
    assert "eval-workspace-candidates" in eval_viewer
    assert "loadEval(\"dir\", defaultPath" in eval_viewer
    assert "trajectory_count" in eval_viewer
    assert ".eval-workspace-candidates" in css
    assert ".eval-candidate-card" in css


def test_eval_viewer_uses_workspace_layout_not_report_stack():
    eval_viewer = read_component("web/components/eval-viewer-client.js")
    css = read_component("web/app/globals.css")

    assert "eval-workspace-shell" in eval_viewer
    assert "eval-workspace-header" in eval_viewer
    assert "eval-workspace-grid" in eval_viewer
    assert "eval-workspace-rail" in eval_viewer
    assert "eval-workspace-viewer" in eval_viewer
    assert "eval-workspace-inspector" in eval_viewer
    assert "eval-workspace-mode-button" in eval_viewer
    assert "eval-workspace-chart-grid" in eval_viewer
    assert "eval-workspace-dim-grid" in eval_viewer
    assert "SimpleTabs" not in eval_viewer
    assert ".eval-workspace-grid" in css
    assert "grid-template-columns: 260px minmax(0, 1fr) 320px;" in css


def test_latency_page_is_not_a_top_level_workspace_column():
    top_nav = read_component("web/components/top-nav.js")
    home = read_component("web/components/home-page-client.js")
    runs = read_component("web/components/runs-page-client.js")

    assert '{ href: "/latency"' not in top_nav
    assert 'href="/latency"' not in home
    assert 'href="/latency"' not in runs
    assert "时延分析" not in home
    assert "时延对比" not in home
    assert "去做时延对比" not in runs


def test_replay_workspace_removes_header_status_and_entity_focus():
    replay = read_component("web/components/run-replay-client.js")
    inspector = read_component("web/components/run-workspace-inspector.js")

    assert "runStatusText" not in replay
    assert "Index ${" not in replay
    assert "Window ${" not in replay
    assert "Entity Focus" not in inspector
    assert "entityFocus" not in replay
    assert "onEntityFocusChange" not in inspector


def test_replay_workspace_uses_tighter_left_rail_layout():
    css = read_component("web/app/globals.css")

    assert "grid-template-columns: 196px minmax(620px, 1fr) 320px;" in css
    assert ".run-workspace-rail {\n  display: grid;\n  gap: 10px;\n  padding: 10px;" in css


def test_replay_workspace_moves_step_summary_to_rail_and_bottom_dock():
    replay = read_component("web/components/run-replay-client.js")
    inspector = read_component("web/components/run-workspace-inspector.js")
    viewer = read_component("web/components/run-workspace-viewer.js")
    css = read_component("web/app/globals.css")

    assert "summary={summary}" in replay
    assert "currentStep={currentStep}" in replay
    assert "stepIdx={stepIdx}" in replay
    assert "maxStep={maxStep}" in replay
    assert "RunWorkspaceRail" in inspector
    assert "Step Summary" in inspector
    assert "Run Summary" in inspector
    assert "run-workspace-rail-grid" in inspector
    assert "run-workspace-data-dock" in viewer
    assert "Current State" in viewer
    assert "First Action" in viewer
    assert "Latency" in viewer
    assert "run-workspace-context-band" not in css


def test_analyze_mode_uses_dedicated_analysis_surface_not_replay_observation():
    viewer = read_component("web/components/run-workspace-viewer.js")
    inspector = read_component("web/components/run-workspace-inspector.js")
    replay_branch = viewer[viewer.index('mode === "replay"') : viewer.index('mode === "attention"')]

    assert 'mode === "analyze" && children' in viewer
    assert 'mode !== "replay" ? (' not in viewer
    assert '<ImageGallery' in replay_branch
    assert "showTrajectory={false}" in read_component("web/components/run-replay-client.js")
    assert 'mode === "analyze" ? (' not in inspector


def test_attention_mode_uses_center_workbench_not_right_heavy_panel():
    viewer = read_component("web/components/run-workspace-viewer.js")
    inspector = read_component("web/components/run-workspace-inspector.js")
    replay = read_component("web/components/run-replay-client.js")
    css = read_component("web/app/globals.css")

    assert "RunWorkspaceAttentionWorkbench" in viewer
    assert "run-workspace-attention-workbench" in viewer
    assert "attentionControls" in viewer
    assert "Generate Step" in viewer
    assert "attentionControls={attentionControls}" in replay
    assert "AttentionPanel" not in inspector
    assert 'mode === "attention" ? (' not in inspector
    assert ".run-workspace-attention-workbench" in css
    assert ".run-workspace-attention-control-panel" in css


def test_analyze_mode_uses_compact_dashboard_grid():
    replay = read_component("web/components/run-replay-client.js")
    css = read_component("web/app/globals.css")
    analyze = replay[replay.index("const analyzeContent") : replay.index("return (", replay.index("const analyzeContent"))]

    assert "run-workspace-analysis-hero" in analyze
    assert "run-workspace-analysis-kpis" in analyze
    assert "run-workspace-analysis-grid-top" in analyze
    assert "run-workspace-chart-grid" in analyze
    assert "FrameDiagnosisPanel" not in analyze
    assert "XYZ first action" not in analyze
    assert "Action magnitude" not in analyze
    assert "Action Stats" not in analyze
    assert ".run-workspace-chart-grid" in css
    assert ".run-workspace-analysis-hero" in css


def test_attention_mode_is_single_step_without_cache_controls():
    viewer = read_component("web/components/run-workspace-viewer.js")
    inspector = read_component("web/components/run-workspace-inspector.js")
    replay = read_component("web/components/run-replay-client.js")
    css = read_component("web/app/globals.css")

    assert "Generate Step" in viewer
    assert "Generate Window" not in viewer
    assert "Refresh" not in viewer
    assert "Step cache" not in viewer
    assert "Cached steps" not in viewer
    assert "Cached" not in viewer
    assert "Uncached" not in viewer
    assert "Attention cache" not in inspector
    assert "attentionCachedSteps" not in inspector
    assert "attentionCachedSteps" not in replay
    assert "onGenerateWindow" not in viewer
    assert "onRefresh" not in viewer
    assert "onGenerateWindow" not in replay
    assert "onRefresh" not in replay
    assert "grid-template-columns: 1fr;" in css[css.index(".run-workspace-attention-grid") : css.index(".run-workspace-attention-card")]


def test_run_workspace_removes_config_mode():
    inspector = read_component("web/components/run-workspace-inspector.js")
    replay = read_component("web/components/run-replay-client.js")

    assert '["config", "Config"]' not in inspector
    assert "Config" not in inspector
    assert 'mode === "config"' not in inspector
    assert "DeletePanel" not in inspector
    assert "deleteControls" not in inspector
    assert 'activeTab === "model"' not in replay
    assert '"config"' not in replay
    assert "deleteArmed" not in replay
    assert "handleDelete" not in replay


def test_open_in_rerun_waits_for_ready_server_before_navigation():
    replay = read_component("web/components/run-replay-client.js")

    assert "Rerun viewer is still starting" in replay
    assert "status.server?.ready === false" in replay
