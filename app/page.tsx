import { Workbench } from "../components/workbench/workbench";
import { runWorkbench } from "../lib/python-bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export default async function Page() {
  const initialState = await runWorkbench({
    job: "simulate",
    tickers: "AAPL",
    source: "auto",
  });

  return (
    <main>
      <Workbench initialState={initialState} />
    </main>
  );
}
