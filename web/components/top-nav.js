import Link from "next/link";

const NAV_ITEMS = [
  { href: "/", label: "Overview" },
  { href: "/runs", label: "Runs" },
  { href: "/latency", label: "Latency" },
  { href: "/datasets", label: "Datasets" },
  { href: "/eval", label: "Eval" },
];

export default function TopNav() {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">VLA-Lab Web</p>
        <Link href="/" className="brand-mark">
          Vision-Language-Action Operations
        </Link>
      </div>
      <nav className="topnav">
        {NAV_ITEMS.map((item) => (
          <Link key={item.href} href={item.href} className="nav-link">
            {item.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
