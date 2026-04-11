# Deploy

## Vercel

For local preview with the same Flask detection Vercel uses in deployment:

```bash
vercel dev
```

To ship it:

```bash
vercel
```

## Fly.io

When you want a container host instead of serverless Flask:

```bash
fly launch --no-deploy
fly deploy
```
