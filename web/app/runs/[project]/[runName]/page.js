import HydrationGuard from "@/components/hydration-guard";
import RunDetailPageClient from "@/components/run-detail-page-client";

export const dynamic = "force-dynamic";

export default async function RunDetailPage({ params }) {
  const route = await params;

  return (
    <HydrationGuard label="正在加载索引化回放工作区 ...">
      <RunDetailPageClient project={route.project} runName={route.runName} />
    </HydrationGuard>
  );
}
