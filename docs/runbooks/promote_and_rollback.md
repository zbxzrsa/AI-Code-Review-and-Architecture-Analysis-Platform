# Channel Promotion and Rollback Guide

## Overview

This guide covers the promotion process between AI channels and rollback procedures.

## Channel Architecture

- **stable**: Production-ready, LTS channel with human approval required
- **next**: Experimental channel with auto-merge on green CI
- **legacy**: Conservative baseline channel

## Promotion Process

### Next → Stable Promotion

**Automatic Promotion** (hourly):

```bash
# Triggered by .github/workflows/promote.yml
# Gates:
- CI green on next branch
- AI eval score_next ≥ eval_score_stable - 0.02
- Last 24h health check passed
```

**Manual Promotion**:

```bash
# Create PR from next to stable
gh pr create -B stable -H next -t "Manual promotion to stable" -b "Manual promotion after review"
```

### Stable → Next/Legacy Backport

**Automatic Backport** (on stable merge):

```bash
# Triggered by .github/workflows/backport.yml
# Creates PRs to next and legacy branches
```

## Rollback Procedures

### Emergency Rollback

**If stable channel issues detected**:

```bash
# 1. Identify problematic commit
git log --oneline stable

# 2. Create rollback branch
git checkout -b rollback-stable <previous-good-commit>

# 3. Push rollback branch
git push origin rollback-stable

# 4. Create emergency PR
gh pr create -B stable -H rollback-stable -t "Emergency rollback" -b "Rollback due to production issue"
```

### Channel-Specific Rollbacks

**Next Channel**:

```bash
# Revert problematic changes
git revert <problematic-commit> --no-edit
git push origin next
```

**Legacy Channel**:

```bash
# Legacy channel changes rarely need rollback
# If needed, follow standard revert process
```

## Health Monitoring

### Check Channel Health

```bash
# Check eval scores
python -m backend.ai.eval --channel stable --dataset datasets/ai_eval/base.jsonl
python -m backend.ai.eval --channel next --dataset datasets/ai_eval/base.jsonl

# Check service health
curl -f http://localhost:8000/healthz
curl -f http://localhost:8000/metrics
```

### Monitor Key Metrics

- `ai_eval_score{channel="stable"}`: Should be ≥ 0.8
- `ai_eval_score{channel="next"}`: Should be within 0.02 of stable
- `ai_requests_total{status="5xx"}`: Should be < 1% of total requests
- `ai_latency_seconds`: 95th percentile should be < 30s

## Troubleshooting

### Promotion Blocked

```bash
# Check CI status
gh run list --branch=next

# Check eval scores
python -m backend.ai.eval --channel next --dataset datasets/ai_eval/base.jsonl --offline

# Check health metrics
curl -s http://localhost:8000/metrics | grep ai_eval_score
```

### Rollback Required

```bash
# Identify issue
docker compose logs api | grep ERROR

# Check recent changes
git log --oneline -10 stable

# Perform rollback (see Emergency Rollback section)
```

## Best Practices

1. **Test promotions in staging first**
2. **Monitor eval scores closely**
3. **Keep rollback commits clean**
4. **Document reasons for rollbacks**
5. **Use feature flags for risky changes**

## Escalation

**Critical Issues**:

1. Create emergency rollback PR
2. Notify on-call engineering team
3. Update status page
4. Document incident in runbook

**Non-Critical Issues**:

1. Create standard rollback PR
2. Schedule team discussion
3. Plan fix for future release
