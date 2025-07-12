"""
清理Dolly数据集中的格式噪声
"""

import os
import re
from datasets import load_dataset
from tqdm import tqdm

def clean_text(text):
    """清理文本中的过度格式化"""
    # 移除连续的###标记
    text = re.sub(r'(###\s*)+', '### ', text)
    
    # 移除过多的换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 移除重复的标题模式
    lines = text.split('\n')
    cleaned_lines = []
    prev_line = ""
    
    for line in lines:
        # 跳过与上一行相同的标题行
        if line.strip().startswith('###') and line.strip() == prev_line.strip():
            continue
        cleaned_lines.append(line)
        prev_line = line
    
    return '\n'.join(cleaned_lines)

def analyze_format_issues():
    """分析数据集中的格式问题"""
    print("加载Dolly数据集...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    format_issues = {
        'excessive_hashtags': 0,
        'repeated_headers': 0,
        'empty_responses': 0,
        'total_samples': len(ds)
    }
    
    for sample in tqdm(ds):
        response = sample['response']
        
        # 检查过多的###
        if response.count('###') > 3:
            format_issues['excessive_hashtags'] += 1
            
        # 检查重复的标题
        lines = response.split('\n')
        headers = [l for l in lines if l.strip().startswith('###')]
        if len(headers) > len(set(headers)):
            format_issues['repeated_headers'] += 1
            
        # 检查空响应
        if len(response.strip()) < 20:
            format_issues['empty_responses'] += 1
    
    print("\n格式问题统计:")
    for key, value in format_issues.items():
        if key != 'total_samples':
            percentage = (value / format_issues['total_samples']) * 100
            print(f"{key}: {value} ({percentage:.2f}%)")

if __name__ == "__main__":
    # 首先分析问题
    analyze_format_issues()
    
    # 询问是否继续清理
    response = input("\n是否继续清理数据? (y/n): ")
    if response.lower() == 'y':
        print("\n开始清理数据...")
        # TODO: 实现完整的清理流程