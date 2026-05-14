# Deployment Notes

## Vercel

The browser UI is configured for Vercel's Python/Flask runtime. Vercel detects
`app.py` as the WSGI entrypoint because it exports a top-level `app` object,
installs runtime dependencies from `requirements.txt`, and reads the Python
version from `.python-version`.

The deployment config intentionally avoids legacy `builds` and uses the modern
`functions` block in `vercel.json` to keep the Python bundle lean while still
including `sample_data/**` for the instant demo.

Local preview:

```bash
python3 -m pip install -r requirements-ui.txt
vercel dev
```

Production deploy:

```bash
vercel deploy --prod
```

Operational notes:

- `Demo sample` is the safest public default because it is deterministic and
  does not depend on live market-data availability.
- `Try live data` can be slower because it calls Yahoo Finance through
  `yfinance`; if the upstream request fails, the app falls back to bundled CSVs.
- Local CSV paths only make sense on machines that have those files. On Vercel,
  use the bundled demo data unless you add CSV files to the repository.
- The function timeout is capped in `vercel.json`; keep browser runs small and
  move larger backtests to the CLI.

## Fly.io

If you want to experiment with a long-running container host instead of
serverless Flask:

```bash
fly launch --no-deploy
fly deploy
```
