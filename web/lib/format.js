export function formatNumber(value, digits = 1) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "--";
  }
  return Number(value).toFixed(digits);
}

export function formatMs(value) {
  const formatted = formatNumber(value, value !== null && value < 10 ? 2 : 1);
  return formatted === "--" ? formatted : `${formatted} ms`;
}

export function formatDate(value) {
  if (!value) {
    return "--";
  }

  if (typeof value === "string") {
    const match = value.match(
      /^(\d{4})-(\d{2})-(\d{2})[T\s](\d{2}):(\d{2})/
    );
    if (match) {
      const [, year, month, day, hour, minute] = match;
      return `${year}-${month}-${day} ${hour}:${minute}`;
    }
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toISOString().slice(0, 16).replace("T", " ");
}

export function formatShortText(value, maxLength = 56) {
  if (!value) {
    return "--";
  }
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 1)}…`;
}

export function toArray(value) {
  if (Array.isArray(value)) {
    return value;
  }
  if (typeof value === "string" && value.length > 0) {
    return [value];
  }
  return [];
}
