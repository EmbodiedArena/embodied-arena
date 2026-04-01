#!/usr/bin/env bash
# EmbSpatial-Bench: 依次运行下方 RUN_SCRIPTS 中列出的 scripts/ 内脚本（仅 basename）。
# 约定：本文件与 scripts/ 目录始终同级；本脚本不修改当前工作目录。
# 子脚本的标准输出/错误输出直接打到当前终端。

set +e

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$THIS_DIR/scripts"

# ---------------------------------------------------------------------------
# 在此数组中填写要运行的脚本文件名（仅 scripts/ 下的文件名）
# 可按需增删或调整顺序
# ---------------------------------------------------------------------------
RUN_SCRIPTS=(
  cambrian.sh
  cosmos_reason1_7b.sh
  embodied_brain.sh
  embodied_vlm.sh
  gemini2_5-pro.sh
  gpt-5.2.sh
  internvl3_5.sh
  mimo_embodied.sh
  o3.sh
  pelican_vl.sh
  qwen-vl-max.sh
  qwen2_5-vl.sh
  qwen3-vl.sh
  robobrain2_7b.sh
  rynnbrain_8b.sh
  step3_vl.sh
  thinker.sh
  wall-oss-fast.sh
)

if [[ ! -d "$SCRIPTS_DIR" ]]; then
  echo "[run_eval] 错误: 与 run_eval.sh 同级的 scripts 目录不存在: $SCRIPTS_DIR"
  exit 1
fi

if [[ ${#RUN_SCRIPTS[@]} -eq 0 ]]; then
  echo "[run_eval] 错误: RUN_SCRIPTS 数组为空，请在 run_eval.sh 中填写要运行的脚本名"
  exit 1
fi

echo "========================================"
echo "EmbSpatial-Bench run_eval"
echo "========================================"
echo "run_eval 所在目录: $THIS_DIR"
echo "scripts 目录:      $SCRIPTS_DIR"
echo "当前工作目录:      $(pwd)"
echo ""
echo "即将依次运行以下脚本（共 ${#RUN_SCRIPTS[@]} 个），输出直接显示在当前终端:"
echo "----------------------------------------"
i=1
for f in "${RUN_SCRIPTS[@]}"; do
  echo "  [$i] scripts/$f"
  ((i++)) || true
done
echo "----------------------------------------"
echo ""
echo "[run_eval] 开始批量执行…"
echo ""

ok_list=()
fail_list=()

for f in "${RUN_SCRIPTS[@]}"; do
  path="$SCRIPTS_DIR/$f"
  if [[ ! -f "$path" ]]; then
    echo "========================================"
    echo "[run_eval] 跳过: scripts/$f（文件不存在）"
    echo "========================================"
    fail_list+=("$f (missing)")
    echo ""
    continue
  fi

  echo "========================================"
  echo "[run_eval] 开始运行: scripts/$f"
  echo "========================================"

  bash "$path"
  ec=$?

  echo ""
  if [[ $ec -eq 0 ]]; then
    echo "[run_eval] 结束运行: scripts/$f — 退出码 0（成功）"
    ok_list+=("$f")
  else
    echo "[run_eval] 结束运行: scripts/$f — 退出码 $ec（失败，继续后续脚本）"
    fail_list+=("$f ($ec)")
  fi
  echo ""
done

echo "========================================"
echo "EmbSpatial-Bench run_eval 汇总"
echo "========================================"
echo "成功: ${#ok_list[@]}"
if [[ ${#ok_list[@]} -gt 0 ]]; then
  for x in "${ok_list[@]}"; do
    echo "  - $x"
  done
else
  echo "  （无）"
fi
echo ""
echo "失败或跳过: ${#fail_list[@]}"
if [[ ${#fail_list[@]} -gt 0 ]]; then
  for x in "${fail_list[@]}"; do
    echo "  - $x"
  done
else
  echo "  （无）"
fi
echo "========================================"
echo "[run_eval] 全部任务已结束。"
echo "========================================"

if [[ ${#fail_list[@]} -gt 0 ]]; then
  exit 1
fi
exit 0
