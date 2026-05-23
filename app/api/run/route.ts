import { NextResponse } from "next/server";

import { runWorkbench } from "../../../lib/python-bridge";
import type { WorkbenchRequest } from "../../../lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function POST(request: Request) {
  let body: Partial<WorkbenchRequest> = {};
  try {
    body = (await request.json()) as Partial<WorkbenchRequest>;
  } catch {
    const payload = await runWorkbench();
    return NextResponse.json({
      ...payload,
      error: "Invalid request body.",
      summary: "Invalid request body.",
    });
  }

  const payload = await runWorkbench(body, { origin: request.url });
  return NextResponse.json(payload);
}
