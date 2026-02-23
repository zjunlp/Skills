# Common Helm Chart Upgrade Issues & Solutions

## 1. Image Pull Errors
### Symptoms
- Pods stuck in `Pending` or `ErrImagePull`/`ImagePullBackOff` state
- Events show: `failed to resolve reference`, `image not found`

### Causes & Fixes
- **Bitnami Registry Change (Post-August 2025):** Free tier access moved to `bitnamilegacy/` repositories.
  - **Solution:** Update all image repository fields in values.yaml:
    