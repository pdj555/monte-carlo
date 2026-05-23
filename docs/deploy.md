# Deployment Notes

## Vercel

The browser workbench is a Next.js App Router application. The UI calls
`/api/run`, which runs the same Python decision engine through `ui_bridge.py`.

Local preview:

```bash
python3 -m pip install -e .
npm install
npm run dev
```

Production deploy:

```bash
vercel deploy --prod
```

Operational notes:

- `Live` downloads Yahoo Finance prices through `yfinance`. When that fails, the
  engine falls back to bundled CSVs under `sample_data/`.
- `Live only` requires a successful download and never falls back.
- `Sample` is deterministic and offline — use it for CI and demos without network.
- `CSV` paths are local to the machine running the Node server. On Vercel, use
  bundled sample data unless you add CSV files to the deployment bundle.
- Browser runs should stay small. Use the CLI for large scenario counts and
  longer backtests.
- `vercel.json` installs both npm and Python dependencies because the Next.js
  route delegates computation to the Python engine.

## Local Launcher

`monte-carlo-ui` is a convenience launcher. It expects `npm install` to have
already created `node_modules/`; otherwise it prints the setup command and
exits without starting a server.
