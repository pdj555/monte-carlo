import type { NextConfig } from "next";

import { engineTraceIncludes, engineTraceRoutes } from "./lib/engine-tracing";

const outputFileTracingIncludes = Object.fromEntries(
  engineTraceRoutes.map((route) => [route, [...engineTraceIncludes]]),
);

const nextConfig: NextConfig = {
  poweredByHeader: false,
  outputFileTracingIncludes,
};

export default nextConfig;
