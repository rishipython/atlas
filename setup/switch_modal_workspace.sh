#!/usr/bin/env bash
set -euo pipefail

# One-shot workspace switch + artifact migration for ATLAS.
#
# Usage:
#   ./setup/switch_modal_workspace.sh
#
# Optional overrides:
#   OLD_PROFILE=mlab-org NEW_PROFILE=rishipython ./setup/switch_modal_workspace.sh

MODAL_BIN="${MODAL_BIN:-/Users/rishi/miniconda3/envs/atlas/bin/modal}"
OLD_PROFILE="${OLD_PROFILE:-mlab-org}"
NEW_PROFILE="${NEW_PROFILE:-rishipython}"
REPO_ROOT="${REPO_ROOT:-/Users/rishi/cs288/atlas}"

if [[ ! -x "$MODAL_BIN" ]]; then
  echo "ERROR: modal binary not found: $MODAL_BIN" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/setup/migrate_modal_workspace.sh" ]]; then
  echo "ERROR: migrate script missing at $REPO_ROOT/setup/migrate_modal_workspace.sh" >&2
  exit 1
fi

echo "[1/5] Checking current profiles"
"$MODAL_BIN" profile current || true
"$MODAL_BIN" profile list || true

echo "[2/5] Ensuring destination profile is active: $NEW_PROFILE"
"$MODAL_BIN" profile activate "$NEW_PROFILE"

# Best-effort cleanup for any bad nested dirs from previous failed copy attempts.
BAD_DIRS=(
  "/oe_fft_convolution_nodiff_s42_v1"
  "/oe_fft_convolution_oe_match_s42_v1"
  "/oe_convolve2d_full_fill_oe_match_s42_v1"
  "/oe_affine_transform_2d_oe_match_s42_v1"
)

echo "[3/5] Cleaning potentially bad top-level dirs in destination volume (best effort)"
for d in "${BAD_DIRS[@]}"; do
  if MODAL_PROFILE="$NEW_PROFILE" "$MODAL_BIN" volume ls atlas-openevolve-outputs "$d" >/dev/null 2>&1; then
    echo "  removing $d"
    MODAL_PROFILE="$NEW_PROFILE" "$MODAL_BIN" volume rm -r atlas-openevolve-outputs "$d" || true
  fi
done

echo "[4/5] Running migration old=$OLD_PROFILE -> new=$NEW_PROFILE"
cd "$REPO_ROOT"
OLD_PROFILE="$OLD_PROFILE" NEW_PROFILE="$NEW_PROFILE" ./setup/migrate_modal_workspace.sh

echo "[5/5] Verifying key paths in destination"
MODAL_PROFILE="$NEW_PROFILE" "$MODAL_BIN" volume ls atlas-openevolve-outputs / | sed -n '1,120p'
MODAL_PROFILE="$NEW_PROFILE" "$MODAL_BIN" volume ls atlas-openevolve-outputs /eval | sed -n '1,120p'
MODAL_PROFILE="$NEW_PROFILE" "$MODAL_BIN" volume ls atlas-openevolve-outputs /synth | sed -n '1,120p'
MODAL_PROFILE="$NEW_PROFILE" "$MODAL_BIN" volume ls atlas-models / | sed -n '1,120p'

echo "Done. If you still see connectivity errors, re-run this script; it is safe to retry."
