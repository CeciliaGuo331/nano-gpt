#!/bin/bash
# 方案2：从step 2继续微调

echo "从Step 2检查点继续微调..."
echo "使用更低的学习率和更少的步数"

python -m model.finetune_dolly \
    --pretrained_checkpoint logs_finetune/model_00002.pt \
    --max_steps 2 \
    --max_lr 5e-7 \
    --warmup_steps 1 \
    --checkpoint_interval 1 \
    --eval_interval 1 \
    --generate_interval 1 \
    --log_dir logs_finetune_continued

echo "微调完成！"