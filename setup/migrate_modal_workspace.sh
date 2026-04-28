#!/usr/bin/env bash
set -euo pipefail

# Copy key ATLAS artifacts between Modal workspaces via local staging.
#
# Usage:
#   OLD_PROFILE=mlab-org NEW_PROFILE=rishipython ./setup/migrate_modal_workspace.sh
#
# Optional:
#   MODAL_BIN=/path/to/modal OLD_PROFILE=... NEW_PROFILE=... ./setup/migrate_modal_workspace.sh
#
# Notes:
# - This copies only the artifacts needed for the current project handoff.
# - It intentionally skips atlas-hf-cache (large, reproducible cache).

MODAL_BIN="${MODAL_BIN:-/Users/rishi/miniconda3/envs/atlas/bin/modal}"
OLD_PROFILE="${OLD_PROFILE:-mlab-org}"
NEW_PROFILE="${NEW_PROFILE:-rishipython}"
STAGING_ROOT="${STAGING_ROOT:-/tmp/modal_workspace_migration}"

if [[ ! -x "$MODAL_BIN" ]]; then
  echo "ERROR: modal binary not found/executable at: $MODAL_BIN" >&2
  exit 1
fi

mkdir -p "$STAGING_ROOT"

copy_path() {
  local profile_from="$1"
  local profile_to="$2"
  local volume="$3"
  local remote_path="$4"

  local local_root="$STAGING_ROOT/$volume"
  local local_path="$local_root/$remote_path"
  local base_name
  base_name="$(basename "$remote_path")"
  local flat_local_path="$local_root/$base_name"
  local parent_dir
  parent_dir="$(dirname "$remote_path")"
  local remote_parent="/"
  if [[ "$parent_dir" != "." ]]; then
    remote_parent="/$parent_dir/"
  fi
  mkdir -p "$local_root"
  # Make retries idempotent: clear stale staged copy for this path.
  rm -rf "$local_path" "$flat_local_path"

  echo "\n[copy] $volume:$remote_path ($profile_from -> $profile_to)"
  # Download into a root staging dir so Modal can recreate nested paths.
  MODAL_PROFILE="$profile_from" "$MODAL_BIN" volume get --force "$volume" "$remote_path" "$local_root"

  # Modal may download nested paths either as <local_root>/<remote_path>
  # or flattened as <local_root>/<basename(remote_path)> depending on path.
  if [[ -e "$local_path" ]]; then
    MODAL_PROFILE="$profile_to" "$MODAL_BIN" volume put -f "$volume" "$local_path" "$remote_parent"
  elif [[ -e "$flat_local_path" ]]; then
    MODAL_PROFILE="$profile_to" "$MODAL_BIN" volume put -f "$volume" "$flat_local_path" "$remote_parent"
  else
    echo "ERROR: staged path missing for $volume:$remote_path" >&2
    echo "  expected one of:" >&2
    echo "    $local_path" >&2
    echo "    $flat_local_path" >&2
    return 1
  fi
}

ensure_volume() {
  local profile="$1"
  local volume="$2"
  echo "[ensure] volume $volume in profile $profile"
  MODAL_PROFILE="$profile" "$MODAL_BIN" volume create "$volume" >/dev/null 2>&1 || true
}

# Ensure destination volumes exist.
ensure_volume "$NEW_PROFILE" "atlas-openevolve-outputs"
ensure_volume "$NEW_PROFILE" "atlas-models"

# -------- atlas-openevolve-outputs (key runs/synth/evals) --------
OE_PATHS=(
  "oe_fft_convolution_nodiff_s42_v1"
  "oe_fft_convolution_oe_match_s42_v1"
  "oe_convolve2d_full_fill_oe_match_s42_v1"
  "oe_affine_transform_2d_oe_match_s42_v1"
  "synth/synth_fft_nodiff_all_v2"
  "synth/synth_fft_nodiff_qualitycheck_v1"
  "synth/synth_pairwise_oe_repoconf_s42_all_v1"
  "eval/base_pass40_algotune_evalresultfix_v2"
  "eval/eval_fft_bestspeed_only_v1"
  "eval/eval_fft_twostage_p2_v1"
)

for p in "${OE_PATHS[@]}"; do
  copy_path "$OLD_PROFILE" "$NEW_PROFILE" "atlas-openevolve-outputs" "$p"
done

# -------- atlas-models (adapters) --------
MODEL_PATHS=(
  "atlas_fftconv_bestspeed_only_v1"
  "atlas_fftconv_twostage_corr_p1_v1"
  "atlas_fftconv_twostage_speed_p2_v1"
)

for p in "${MODEL_PATHS[@]}"; do
  copy_path "$OLD_PROFILE" "$NEW_PROFILE" "atlas-models" "$p"
done

echo "\nMigration complete."
echo "Staging kept at: $STAGING_ROOT"
echo "You can remove it later with: rm -rf '$STAGING_ROOT'"
