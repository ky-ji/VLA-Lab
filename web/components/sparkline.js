function normalize(values, width, height, padding) {
  const valid = values
    .map((value, index) => ({ value, index }))
    .filter((item) => item.value !== null && item.value !== undefined && !Number.isNaN(item.value));

  if (valid.length === 0) {
    return "";
  }

  const numbers = valid.map((item) => item.value);
  const min = Math.min(...numbers);
  const max = Math.max(...numbers);
  const span = max - min || 1;
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;

  return valid
    .map((item) => {
      const x = padding + (usableWidth * item.index) / Math.max(values.length - 1, 1);
      const y = height - padding - ((item.value - min) / span) * usableHeight;
      return `${x},${y}`;
    })
    .join(" ");
}

export default function Sparkline({
  values,
  width = 320,
  height = 110,
  stroke = "#0f766e",
  title,
}) {
  const polyline = normalize(values, width, height, 8);

  if (!polyline) {
    return <div className="empty-sparkline">No timing samples</div>;
  }

  return (
    <figure className="sparkline-shell">
      {title ? <figcaption>{title}</figcaption> : null}
      <svg viewBox={`0 0 ${width} ${height}`} className="sparkline">
        <rect x="0" y="0" width={width} height={height} rx="18" fill="rgba(255,255,255,0.7)" />
        <polyline
          fill="none"
          stroke={stroke}
          strokeWidth="3"
          strokeLinejoin="round"
          strokeLinecap="round"
          points={polyline}
        />
      </svg>
    </figure>
  );
}
