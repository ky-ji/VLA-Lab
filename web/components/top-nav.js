"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "总览" },
  { href: "/deploy", label: "部署" },
  { href: "/runs", label: "Runs" },
  { href: "/latency", label: "时延" },
  { href: "/datasets", label: "数据集" },
  { href: "/eval", label: "评估" },
];

function isActive(pathname, href) {
  if (href === "/") return pathname === "/";
  return pathname.startsWith(href);
}

export default function TopNav() {
  const pathname = usePathname();

  return (
    <header className="topbar">
      <Link href="/" className="brand-mark">
        VLA-Lab
      </Link>
      <nav className="topnav">
        {NAV_ITEMS.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`nav-link${isActive(pathname, item.href) ? " is-active" : ""}`}
          >
            {item.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
