# Rollback Procedure

## Quick Rollback

```bash
# Rollback to last working commit
git revert --no-edit HEAD
git push origin main

# Alternative: Force reset to known good commit
git reset --hard 70aa21081f8e4fd1ff6c95c7a33785283bb77d89
git push origin main --force
```

## Vercel Deployment Rollback

1. Go to Vercel Dashboard
2. Select your project
3. Click on "Deployments" tab
4. Find the last successful deployment
5. Click "..." menu and select "Promote to Production"

## Emergency Contacts

- DevOps Lead: [Contact Info]
- Vercel Support: [Contact Info]

## Verification Steps After Rollback

1. [ ] Homepage loads: https://your-domain.vercel.app/
2. [ ] No 404 errors on core routes
3. [ ] API proxy working (if configured)
4. [ ] Static assets loading correctly
