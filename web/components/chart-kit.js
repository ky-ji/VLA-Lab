"use client";

import { useEffect, useRef, useState } from "react";

function coerceNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return null;
  }
  return Number(value);
}

function normalizeSeries(series) {
  return (series || []).map((item) => ({
    ...item,
    values: (item.values || []).map(coerceNumber),
  }));
}

function collectBounds(series) {
  const values = normalizeSeries(series).flatMap((item) => item.values.filter((value) => value !== null));
  if (values.length === 0) {
    return { min: 0, max: 1 };
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    return { min: min - 1, max: max + 1 };
  }
  return { min, max };
}

function buildPolyline(values, width, height, padding, min, max) {
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;
  const span = max - min || 1;

  return values
    .map((value, index) => {
      if (value === null) {
        return null;
      }
      const x = padding + (usableWidth * index) / Math.max(values.length - 1, 1);
      const y = height - padding - ((value - min) / span) * usableHeight;
      return `${x},${y}`;
    })
    .filter(Boolean)
    .join(" ");
}

export function LineChart({
  series = [],
  markerIndex = null,
  title,
  height = 220,
  xLabel,
}) {
  const width = 860;
  const padding = 22;
  const normalized = normalizeSeries(series);
  const maxLength = Math.max(0, ...normalized.map((item) => item.values.length));
  const { min, max } = collectBounds(normalized);
  const markerX =
    markerIndex === null || maxLength <= 1
      ? null
      : padding + ((width - padding * 2) * markerIndex) / Math.max(maxLength - 1, 1);

  if (maxLength === 0) {
    return <div className="chart-empty">No series data</div>;
  }

  return (
    <div className="chart-shell">
      {title ? <div className="chart-title">{title}</div> : null}
      <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg">
        <rect x="0" y="0" width={width} height={height} rx="24" fill="rgba(255,255,255,0.86)" />
        {[0, 1, 2, 3].map((tick) => {
          const y = padding + ((height - padding * 2) * tick) / 3;
          return (
            <line
              key={tick}
              x1={padding}
              y1={y}
              x2={width - padding}
              y2={y}
              stroke="rgba(31, 41, 51, 0.09)"
              strokeDasharray="4 8"
            />
          );
        })}
        {normalized.map((item) => (
          <polyline
            key={item.name}
            fill="none"
            stroke={item.color || "#0f766e"}
            strokeWidth="3"
            strokeLinejoin="round"
            strokeLinecap="round"
            points={buildPolyline(item.values, width, height, padding, min, max)}
          />
        ))}
        {markerX !== null ? (
          <line
            x1={markerX}
            y1={padding}
            x2={markerX}
            y2={height - padding}
            stroke="rgba(15, 23, 42, 0.5)"
            strokeDasharray="6 6"
            strokeWidth="2"
          />
        ) : null}
      </svg>
      <div className="chart-legend">
        {normalized.map((item) => (
          <span key={item.name} className="chart-legend-item">
            <span className="chart-dot" style={{ backgroundColor: item.color || "#0f766e" }} />
            {item.name}
          </span>
        ))}
        {xLabel ? <span className="chart-footnote">{xLabel}</span> : null}
      </div>
    </div>
  );
}

function downsampleHeatmap(matrix, maxCols = 80) {
  if (!matrix || matrix.length === 0) {
    return [];
  }
  const rowCount = matrix.length;
  const colCount = matrix[0]?.length || 0;
  if (colCount <= maxCols) {
    return matrix;
  }
  const stride = Math.ceil(colCount / maxCols);
  return matrix.map((row) => row.filter((_, index) => index % stride === 0));
}

function colorForValue(value, min, max) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "rgba(226, 232, 240, 0.8)";
  }
  const ratio = (Number(value) - min) / Math.max(max - min, 1e-6);
  const clamped = Math.max(0, Math.min(1, ratio));
  const hue = 210 - clamped * 180;
  const light = 92 - clamped * 38;
  return `hsl(${hue}, 78%, ${light}%)`;
}

export function HeatmapChart({ matrix = [], rowLabels = [], title }) {
  const sampled = downsampleHeatmap(matrix);
  const values = sampled.flat().map(coerceNumber).filter((item) => item !== null);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;

  if (!sampled.length || !sampled[0]?.length) {
    return <div className="chart-empty">No heatmap data</div>;
  }

  return (
    <div className="chart-shell">
      {title ? <div className="chart-title">{title}</div> : null}
      <div className="heatmap-shell">
        <div className="heatmap-labels">
          {sampled.map((_, rowIdx) => (
            <span key={rowIdx}>{rowLabels[rowIdx] || rowIdx}</span>
          ))}
        </div>
        <div
          className="heatmap-grid"
          style={{ gridTemplateColumns: `repeat(${sampled[0].length}, minmax(0, 1fr))` }}
        >
          {sampled.flatMap((row, rowIdx) =>
            row.map((value, colIdx) => (
              <span
                key={`${rowIdx}-${colIdx}`}
                className="heatmap-cell"
                style={{ backgroundColor: colorForValue(value, min, max) }}
                title={`${rowLabels[rowIdx] || rowIdx} / ${colIdx}: ${value ?? "--"}`}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export function TrajectoryProjection({ series = [], title, currentPoint = null }) {
  const chartRef = useRef(null);
  const [plotly, setPlotly] = useState(null);
  const [renderError, setRenderError] = useState(null);
  const normalizedSeries = (series || []).map((item) => ({
    ...item,
    points: (item.points || []).map((point) => [
      Number(point?.[0] ?? 0),
      Number(point?.[1] ?? 0),
      Number(point?.[2] ?? 0),
    ]),
  }));
  const hasPoints = normalizedSeries.some((item) => item.points.length > 0) || Boolean(currentPoint?.length);

  useEffect(() => {
    let cancelled = false;
    import("plotly.js-dist-min").then((module) => {
      if (!cancelled) {
        setPlotly(module.default || module);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!plotly || !chartRef.current || !hasPoints) {
      return undefined;
    }

    let active = true;

    const traces = [];
    normalizedSeries.forEach((item) => {
      if (!item.points.length) {
        return;
      }
      const mode = item.mode || "lines";
      const trace = {
        type: "scatter3d",
        mode,
        name: item.name,
        x: item.points.map((point) => point[0]),
        y: item.points.map((point) => point[1]),
        z: item.points.map((point) => point[2]),
        hovertemplate: `${item.name}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>`,
      };

      if (mode.includes("lines")) {
        trace.line = {
          color: item.color || "#0f766e",
          width: 6,
          dash: item.dashed ? "dash" : "solid",
        };
      }

      if (mode.includes("markers")) {
        trace.marker = {
          size: item.markerSize || 3,
          color: item.color || "#0f766e",
          opacity: item.opacity ?? 0.72,
        };
      }

      traces.push(trace);

      if (!mode.includes("lines")) {
        return;
      }

      const firstPoint = item.points[0];
      const lastPoint = item.points[item.points.length - 1];
      traces.push({
        type: "scatter3d",
        mode: "markers",
        showlegend: false,
        x: [firstPoint[0]],
        y: [firstPoint[1]],
        z: [firstPoint[2]],
        marker: {
          size: 4,
          color: item.color || "#0f766e",
          symbol: "diamond",
        },
        hovertemplate: `${item.name} start<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>`,
      });
      traces.push({
        type: "scatter3d",
        mode: "markers",
        showlegend: false,
        x: [lastPoint[0]],
        y: [lastPoint[1]],
        z: [lastPoint[2]],
        marker: {
          size: 4,
          color: item.color || "#0f766e",
          symbol: "circle",
        },
        hovertemplate: `${item.name} end<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>`,
      });
    });

    if (currentPoint?.length) {
      traces.push({
        type: "scatter3d",
        mode: "markers",
        name: "Current",
        x: [Number(currentPoint[0] ?? 0)],
        y: [Number(currentPoint[1] ?? 0)],
        z: [Number(currentPoint[2] ?? 0)],
        marker: {
          size: 7,
          color: "#2563eb",
          line: { color: "#ffffff", width: 2 },
        },
        hovertemplate: "Current<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
      });
    }

    const renderPlot = async () => {
      try {
        await plotly.react(
          chartRef.current,
          traces,
          {
            paper_bgcolor: "rgba(255,255,255,0.86)",
            plot_bgcolor: "rgba(255,255,255,0.86)",
            margin: { l: 0, r: 0, b: 0, t: 0 },
            showlegend: true,
            uirevision: "trajectory",
            legend: {
              orientation: "h",
              y: -0.1,
              x: 0,
            },
            scene: {
              xaxis: { title: "X", backgroundcolor: "rgba(255,255,255,0.0)", gridcolor: "rgba(31,41,51,0.10)" },
              yaxis: { title: "Y", backgroundcolor: "rgba(255,255,255,0.0)", gridcolor: "rgba(31,41,51,0.10)" },
              zaxis: { title: "Z", backgroundcolor: "rgba(255,255,255,0.0)", gridcolor: "rgba(31,41,51,0.10)" },
              camera: {
                eye: { x: 1.45, y: 1.3, z: 0.85 },
              },
              aspectmode: "data",
            },
          },
          {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: [
              "select2d",
              "lasso2d",
              "zoom2d",
              "pan2d",
              "zoomIn2d",
              "zoomOut2d",
              "autoScale2d",
              "toggleSpikelines",
            ],
          }
        );
        if (active) {
          setRenderError(null);
        }
      } catch (error) {
        console.error("Failed to render Plotly trajectory", error);
        if (active) {
          setRenderError(error instanceof Error ? error.message : String(error));
        }
      }
    };

    renderPlot();

    return () => {
      active = false;
    };
  }, [currentPoint, hasPoints, normalizedSeries, plotly]);

  useEffect(() => {
    return () => {
      if (plotly && chartRef.current) {
        plotly.purge(chartRef.current);
      }
    };
  }, [plotly]);

  if (!hasPoints) {
    return <div className="chart-empty">No trajectory data</div>;
  }

  if (renderError) {
    return <div className="chart-empty">Trajectory render failed: {renderError}</div>;
  }

  return (
    <div className="chart-shell">
      {title ? <div className="chart-title">{title}</div> : null}
      <div ref={chartRef} className="plotly-frame" />
    </div>
  );
}

function binsForValues(values, binCount = 18) {
  const valid = values.map(coerceNumber).filter((item) => item !== null);
  if (!valid.length) {
    return [];
  }
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const span = max - min || 1;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    label: min + (span * index) / binCount,
    count: 0,
  }));
  for (const value of valid) {
    const idx = Math.min(binCount - 1, Math.floor(((value - min) / span) * binCount));
    bins[idx].count += 1;
  }
  return bins;
}

export function HistogramChart({ values = [], title, color = "#0f766e" }) {
  const bins = binsForValues(values);
  const maxCount = Math.max(1, ...bins.map((item) => item.count));

  if (!bins.length) {
    return <div className="chart-empty">No histogram data</div>;
  }

  return (
    <div className="chart-shell">
      {title ? <div className="chart-title">{title}</div> : null}
      <div className="histogram-bars">
        {bins.map((bin, index) => (
          <div key={index} className="histogram-bar-wrap">
            <span
              className="histogram-bar"
              style={{
                height: `${(bin.count / maxCount) * 100}%`,
                backgroundColor: color,
              }}
              title={`${bin.label.toFixed(3)} : ${bin.count}`}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export function TimelineEvents({ events = [] }) {
  const width = 760;
  const height = 120;
  const padding = 24;
  const offsets = events.map((item) => Number(item.offset_ms));

  if (!events.length) {
    return <div className="chart-empty">无时序事件</div>;
  }

  const min = Math.min(...offsets);
  const max = Math.max(...offsets);
  const span = max - min || 1;
  const mapX = (offset) => padding + ((offset - min) / span) * (width - padding * 2);

  return (
    <div className="chart-shell">
      <div className="chart-title">帧时间线</div>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg">
        <rect x="0" y="0" width={width} height={height} rx="24" fill="rgba(255,255,255,0.86)" />
        <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="rgba(31,41,51,0.24)" strokeWidth="2" />
        {events.map((event, index) => {
          const x = mapX(Number(event.offset_ms));
          const yOffset = index % 2 === 0 ? -28 : 28;
          return (
            <g key={`${event.label}-${index}`}>
              <circle cx={x} cy={height / 2} r="8" fill={event.color || "#0f766e"} />
              <text x={x} y={height / 2 + yOffset} textAnchor="middle" className="timeline-label">
                {event.label}
              </text>
              <text x={x} y={height / 2 + yOffset + 16} textAnchor="middle" className="timeline-value">
                {Number(event.offset_ms).toFixed(1)} ms
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function SimpleTabs({ tabs = [], activeTab, onChange }) {
  return (
    <div className="tab-strip">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          type="button"
          className={`tab-pill${tab.key === activeTab ? " is-active" : ""}`}
          onClick={() => onChange(tab.key)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
