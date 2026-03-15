import HydrationGuard from "@/components/hydration-guard";
import LatencyCompareClient from "@/components/latency-compare-client";
import { getLatencyCompare, getRuns } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function LatencyPage() {
  const runsResponse = await getRuns({
    include_latency: true,
    limit: 240,
  });
  const initialRuns = runsResponse?.items || [];
  const initialRunIds = initialRuns.slice(0, 3).map((run) => run.run_id);
  const initialCompare = initialRunIds.length ? await getLatencyCompare(initialRunIds) : { items: [] };

  return (
    <HydrationGuard label="正在加载时延分析 ...">
      <LatencyCompareClient initialRuns={initialRuns} initialCompare={initialCompare} />
    </HydrationGuard>
  );
}
