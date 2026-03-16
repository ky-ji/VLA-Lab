import DeployDashboardClient from "@/components/deploy-dashboard-client";
import { getDeployOverview } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function DeployPage() {
  const overview = await getDeployOverview();

  return <DeployDashboardClient initialOverview={overview} />;
}
