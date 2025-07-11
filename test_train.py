"""
快速测试train_gpt2.py的脚本
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量用于测试
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一个GPU

# 导入训练脚本
from model import train_gpt2

# 修改一些参数以加快测试
train_gpt2.max_steps = 5  # 只训练100步
train_gpt2.B = 2  # 减小batch size
train_gpt2.T = 256  # 减小序列长度
train_gpt2.total_batch_size = 4096  # 减小总batch size

print("开始测试训练...")
print(f"最大步数: {train_gpt2.max_steps}")
print(f"Batch size: {train_gpt2.B}")
print(f"序列长度: {train_gpt2.T}")
