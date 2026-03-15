import "./globals.css";

import TopNav from "@/components/top-nav";

export const metadata = {
  title: "VLA-Lab",
  description: "VLA 模型部署运维可视化工作台",
};

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN">
      <body suppressHydrationWarning>
        <div className="app-shell">
          <TopNav />
          <main className="content-shell">{children}</main>
        </div>
      </body>
    </html>
  );
}
