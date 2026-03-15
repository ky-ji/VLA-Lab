"use client";

import { useEffect, useState } from "react";

export default function HydrationGuard({ children, label = "正在加载交互界面..." }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <section className="placeholder-panel">
        <p>{label}</p>
      </section>
    );
  }

  return children;
}
