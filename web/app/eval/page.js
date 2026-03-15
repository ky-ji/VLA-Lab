import EvalViewerClient from "@/components/eval-viewer-client";
import HydrationGuard from "@/components/hydration-guard";

export const dynamic = "force-dynamic";

export default function EvalPage() {
  return (
    <HydrationGuard label="正在加载评估界面 ...">
      <EvalViewerClient />
    </HydrationGuard>
  );
}
