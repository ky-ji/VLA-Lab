import "./globals.css";

import TopNav from "@/components/top-nav";

export const metadata = {
  title: "VLA-Lab Web",
  description: "FastAPI + Next.js dashboard for VLA deployment logs.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="app-shell">
          <TopNav />
          <main className="content-shell">{children}</main>
        </div>
      </body>
    </html>
  );
}
