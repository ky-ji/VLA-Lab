import DatasetViewerClient from "@/components/dataset-viewer-client";
import HydrationGuard from "@/components/hydration-guard";

export const dynamic = "force-dynamic";

export default function DatasetsPage() {
  return (
    <HydrationGuard label="正在加载数据集交互界面...">
      <DatasetViewerClient />
    </HydrationGuard>
  );
}
