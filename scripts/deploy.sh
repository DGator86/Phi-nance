#!/bin/bash
# Phi-nance master deploy script
# Handles: abort stale rebase → reset local MAIN to origin → merge feature → push
# Usage: ./scripts/deploy.sh [--dry-run] [--feature <branch>]
set -euo pipefail
cd "$(dirname "$0")/.."

MAIN_BRANCH="MAIN"
FEATURE_BRANCH="claude/review-paperclip-integration-4HLI1"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --feature) shift; FEATURE_BRANCH="${1:?--feature requires a branch name}" ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

# ── colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[info]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}   $*"; }
error()   { echo -e "${RED}[error]${NC}  $*"; exit 1; }
section() { echo -e "\n${CYAN}══ $* ══${NC}"; }

# ── 1. abort any in-progress git operations ────────────────────────────────────
section "Checking for in-progress git operations"

if [[ -d ".git/rebase-merge" || -d ".git/rebase-apply" ]]; then
  warn "Rebase in progress — aborting it."
  $DRY_RUN || git rebase --abort
  info "Rebase aborted."
else
  info "No in-progress rebase found."
fi

if [[ -f ".git/MERGE_HEAD" ]]; then
  warn "Merge in progress — aborting it."
  $DRY_RUN || git merge --abort
  info "Merge aborted."
fi

if [[ -f ".git/CHERRY_PICK_HEAD" ]]; then
  warn "Cherry-pick in progress — aborting it."
  $DRY_RUN || git cherry-pick --abort
  info "Cherry-pick aborted."
fi

# ── 2. fetch everything ────────────────────────────────────────────────────────
section "Fetching origin"
$DRY_RUN || git fetch origin
info "Fetch complete."

# ── 3. reset local MAIN to match origin/MAIN exactly ──────────────────────────
section "Resetting local $MAIN_BRANCH to origin/$MAIN_BRANCH"

LOCAL_AHEAD=$(git rev-list --count "origin/$MAIN_BRANCH..refs/heads/$MAIN_BRANCH" 2>/dev/null || echo 0)
LOCAL_BEHIND=$(git rev-list --count "refs/heads/$MAIN_BRANCH..origin/$MAIN_BRANCH" 2>/dev/null || echo 0)

info "Local $MAIN_BRANCH is ${LOCAL_AHEAD} ahead / ${LOCAL_BEHIND} behind origin/$MAIN_BRANCH."

if [[ "$LOCAL_AHEAD" -gt 0 || "$LOCAL_BEHIND" -gt 0 ]]; then
  warn "Local $MAIN_BRANCH has diverged — resetting to origin/$MAIN_BRANCH."
  if ! $DRY_RUN; then
    git checkout "$MAIN_BRANCH" 2>/dev/null || git checkout -b "$MAIN_BRANCH" "origin/$MAIN_BRANCH"
    git reset --hard "origin/$MAIN_BRANCH"
  fi
  info "Reset complete."
else
  info "Local $MAIN_BRANCH is clean — no reset needed."
  $DRY_RUN || git checkout "$MAIN_BRANCH"
fi

# ── 4. show what will be merged ────────────────────────────────────────────────
section "Commits in $FEATURE_BRANCH not yet in $MAIN_BRANCH"

COMMITS_AHEAD=$(git rev-list --count "$MAIN_BRANCH..$FEATURE_BRANCH" 2>/dev/null || echo 0)

if [[ "$COMMITS_AHEAD" -eq 0 ]]; then
  info "$MAIN_BRANCH is already up to date with $FEATURE_BRANCH — nothing to merge."
  exit 0
fi

info "$FEATURE_BRANCH has $COMMITS_AHEAD commit(s) to merge:"
git --no-pager log --oneline "$MAIN_BRANCH..$FEATURE_BRANCH"
echo ""

$DRY_RUN && { warn "Dry run — stopping before merge."; exit 0; }

# ── 5. merge feature → MAIN ────────────────────────────────────────────────────
section "Merging $FEATURE_BRANCH → $MAIN_BRANCH"

if git merge --ff-only "$FEATURE_BRANCH"; then
  info "Fast-forward merge successful."
else
  warn "Fast-forward not possible — creating merge commit."
  git merge --no-ff "$FEATURE_BRANCH" -m "Merge branch '$FEATURE_BRANCH' into $MAIN_BRANCH"
  info "Merge commit created."
fi

# ── 6. push with retry ─────────────────────────────────────────────────────────
section "Pushing $MAIN_BRANCH to origin"

ATTEMPT=0
WAIT=2
while true; do
  if git push -u origin "$MAIN_BRANCH"; then
    info "Push successful."
    break
  fi
  ATTEMPT=$((ATTEMPT + 1))
  [[ $ATTEMPT -ge 4 ]] && error "Push failed after 4 attempts — check credentials/network."
  warn "Push failed — retrying in ${WAIT}s (attempt $ATTEMPT/4)..."
  sleep "$WAIT"
  WAIT=$((WAIT * 2))
done

# ── 7. summary ─────────────────────────────────────────────────────────────────
section "Done"
info "$MAIN_BRANCH is now up to date on origin."
echo ""
echo "  Latest 5 commits on $MAIN_BRANCH:"
git --no-pager log --oneline -5
echo ""
echo "  App VPS startup (once SSH is reachable):"
echo "    cd ~/Phi-nance && git pull origin MAIN"
echo "    source venv/bin/activate"
echo "    screen -r phi-nance  # Ctrl+C to stop"
echo "    ./start.sh           # Ctrl+A then D to detach"
