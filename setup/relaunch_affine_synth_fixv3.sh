#!/usr/bin/env bash
set -euo pipefail
MODAL_PROFILE=${MODAL_PROFILE:-rishipython}
MODAL=/Users/rishi/miniconda3/envs/atlas/bin/modal
ROOT=/Users/rishi/cs288/atlas
OUT="$ROOT/runs/stream_perm_v1_auto_chain/synth_fixv3"
mkdir -p "$OUT"
APPS="$OUT/apps.tsv"
: > "$APPS"

launch() {
  local method="$1" shard="$2" outname="$3"
  local trace="$ROOT/runs/stream_perm_v1_auto_chain/$method/affine_transform_2d/synth_fixv2_shards/shard_${shard}.jsonl"
  local log="$OUT/${method}_sh${shard}.launch.log"
  local txt
  txt=$(MODAL_PROFILE="$MODAL_PROFILE" "$MODAL" run -d experiment/synth_reasoning.py::main_algotune --trace-path "$trace" --problem-id affine_transform_2d --out-name "$outname" 2>&1 | tee "$log")
  local app_id
  app_id=$(printf '%s\n' "$txt" | sed -n 's#.*\(ap-[A-Za-z0-9]\+\).*#\1#p' | tail -n1)
  if [[ -z "$app_id" ]]; then
    echo "FAILED_PARSE_APP_ID $method sh$shard" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\n' "$method" "$shard" "$outname" "$app_id" >> "$APPS"
  echo "launched $method sh$shard => $app_id"
}

launch sft-best-traj 0 stream_perm_v1_sft-best-traj_affine_transform_2d_synth_fixv3_sh0
launch sft-best-traj 1 stream_perm_v1_sft-best-traj_affine_transform_2d_synth_fixv3_sh1
launch sft-best-traj 2 stream_perm_v1_sft-best-traj_affine_transform_2d_synth_fixv3_sh2
launch sft-best-traj 3 stream_perm_v1_sft-best-traj_affine_transform_2d_synth_fixv3_sh3
launch dpo-weighted-sft 0 stream_perm_v1_dpo-weighted-sft_affine_transform_2d_synth_fixv3_sh0
launch dpo-weighted-sft 1 stream_perm_v1_dpo-weighted-sft_affine_transform_2d_synth_fixv3_sh1
launch dpo-weighted-sft 2 stream_perm_v1_dpo-weighted-sft_affine_transform_2d_synth_fixv3_sh2
launch rlm 0 stream_perm_v1_rlm_affine_transform_2d_synth_fixv3_sh0
launch rlm 1 stream_perm_v1_rlm_affine_transform_2d_synth_fixv3_sh1
launch rlm 2 stream_perm_v1_rlm_affine_transform_2d_synth_fixv3_sh2

echo "apps written to $APPS"
