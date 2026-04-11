# Deployment Notes

These are operator notes from manual experiments. This repository does not yet
ship maintained deployment config or CI coverage for Vercel or Fly.io.

## Vercel

For local preview with the same Flask detection Vercel uses:

```bash
vercel dev
```

If you want to experiment with deployment:

```bash
vercel
```

## Fly.io

If you want to experiment with a container host instead of serverless Flask:

```bash
fly launch --no-deploy
fly deploy
```
