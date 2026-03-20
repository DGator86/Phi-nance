#!/bin/bash
# Phi-nance: merge current feature branch into MAIN and push
# Usage: ./scripts/merge_to_main.sh [--dry-run]
set -euo pipefail

MAIN_BRANCH="MAIN"
FEATURE_BRANCH=$(git rev-parse --abbrev-ref HEAD)
DRY_RUN=false

for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

# ── colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[info]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── preflight (ignore untracked files — only block on staged/modified tracked files) ──
[[ -n "$(git status --porcelain | grep -v '^??')" ]] && error "Staged or modified tracked files present — commit or stash first."

info "Feature branch : $FEATURE_BRANCH"
info "Target branch  : $MAIN_BRANCH"

# ── fetch latest state ─────────────────────────────────────────────────────────
info "Fetching origin..."
git fetch origin

# ── fast-forward check ─────────────────────────────────────────────────────────
AHEAD=$(git rev-list --count "origin/$MAIN_BRANCH..HEAD" 2>/dev/null || echo 0)
BEHIND=$(git rev-list --count "HEAD..origin/$MAIN_BRANCH" 2>/dev/null || echo 0)

info "Feature is ${AHEAD} commit(s) ahead, ${BEHIND} commit(s) behind $MAIN_BRANCH."

if [[ "$BEHIND" -gt 0 ]]; then
  warn "$FEATURE_BRANCH is behind $MAIN_BRANCH — will rebase first to keep history clean."
  $DRY_RUN || git rebase "origin/$MAIN_BRANCH"
fi

if [[ "$AHEAD" -eq 0 ]]; then
  info "Nothing to merge — $MAIN_BRANCH is already up to date."
  exit 0
fi

# ── preview what will be merged ────────────────────────────────────────────────
echo ""
echo "Commits to merge into $MAIN_BRANCH:"
git log --oneline "origin/$MAIN_BRANCH..HEAD"
echo ""

$DRY_RUN && { warn "Dry run — stopping here."; exit 0; }

# ── switch to MAIN, pull, merge ────────────────────────────────────────────────
info "Switching to $MAIN_BRANCH..."
git checkout "$MAIN_BRANCH" 2>/dev/null || git checkout -b "$MAIN_BRANCH" "origin/$MAIN_BRANCH"

info "Pulling latest $MAIN_BRANCH..."
git pull --ff-only origin "$MAIN_BRANCH" || true

info "Merging $FEATURE_BRANCH → $MAIN_BRANCH (fast-forward)..."
if ! git merge --ff-only "$FEATURE_BRANCH"; then
  warn "Fast-forward not possible — falling back to merge commit."
  git merge --no-ff "$FEATURE_BRANCH" -m "Merge branch '$FEATURE_BRANCH' into $MAIN_BRANCH"
fi

# ── push ───────────────────────────────────────────────────────────────────────
info "Pushing $MAIN_BRANCH to origin..."
ATTEMPT=0
WAIT=2
while true; do
  if git push -u origin "$MAIN_BRANCH"; then
    break
  fi
  ATTEMPT=$((ATTEMPT + 1))
  [[ $ATTEMPT -ge 4 ]] && error "Push failed after 4 attempts."
  warn "Push failed — retrying in ${WAIT}s (attempt $ATTEMPT/4)..."
  sleep "$WAIT"
  WAIT=$((WAIT * 2))
done

# ── done ───────────────────────────────────────────────────────────────────────
info "Done. $MAIN_BRANCH is now up to date."
echo ""
echo "Next: on the VPS run:"
echo "  cd ~/Phi-nance && git pull origin MAIN"
echo "  source venv/bin/activate"
echo "  screen -r phi-nance   # Ctrl+C, then:"
echo "  ./start.sh"
