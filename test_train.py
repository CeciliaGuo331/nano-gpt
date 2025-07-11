"""
快速测试train_gpt2.py的所有功能
包括：基本训练、checkpoint保存/加载、断点续训、自动恢复等
"""

import os
import sys
import time
import subprocess
import shutil
import argparse

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量用于测试
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一个GPU

def run_command(cmd, test_name):
    """运行命令并捕获输出"""
    print(f"\n{'='*60}")
    print(f"测试: {test_name}")
    print(f"命令: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✓ 成功 (耗时: {elapsed:.2f}秒)")
    else:
        print(f"✗ 失败 (耗时: {elapsed:.2f}秒)")
        print(f"错误输出: {result.stderr}")
    
    return result.returncode == 0

def modify_train_script():
    """临时修改训练脚本参数以加快测试"""
    # 读取原始脚本
    train_script_path = "model/train_gpt2.py"
    with open(train_script_path, 'r') as f:
        content = f.read()
    
    # 备份原始脚本
    backup_path = train_script_path + ".backup"
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # 修改参数
    modifications = {
        "max_steps = 19073": "max_steps = 10  # 测试用",
        "B = 16": "B = 4  # 测试用",
        "T = 1024": "T = 256  # 测试用",
        "total_batch_size = 524288": "total_batch_size = 1024  # 测试用",
        "val_loss_steps = 20": "val_loss_steps = 2  # 测试用",
        "if step % 250 == 0": "if step % 2 == 0  # 测试用",
        "warmup_steps = 715": "warmup_steps = 2  # 测试用",
    }
    
    for old, new in modifications.items():
        content = content.replace(old, new)
    
    # 添加sys.path修复
    import_fix = """import sys
import os
# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
    content = import_fix + content
    
    # 写回修改后的脚本
    with open(train_script_path, 'w') as f:
        f.write(content)
    
    return backup_path

def restore_train_script(backup_path):
    """恢复原始训练脚本"""
    train_script_path = "model/train_gpt2.py"
    shutil.move(backup_path, train_script_path)

def clean_test_env():
    """清理测试环境"""
    # 备份现有的log目录
    if os.path.exists("log"):
        backup_dir = f"log_backup_{int(time.time())}"
        shutil.move("log", backup_dir)
        print(f"已备份原log目录到: {backup_dir}")
    
    # 创建新的log目录
    os.makedirs("log", exist_ok=True)

def test_basic_training():
    """测试基本训练功能"""
    cmd = "python model/train_gpt2.py"
    return run_command(cmd, "基本训练功能")

def test_checkpoint_save():
    """测试checkpoint保存功能"""
    cmd = "python model/train_gpt2.py --checkpoint_interval 5"
    success = run_command(cmd, "Checkpoint保存功能")
    
    if success:
        # 检查是否生成了checkpoint文件
        checkpoints = [f for f in os.listdir("log") if f.startswith("model_") and f.endswith(".pt")]
        print(f"生成的checkpoint文件: {checkpoints}")
        return len(checkpoints) > 0
    return False

def test_resume_from_checkpoint():
    """测试从指定checkpoint恢复"""
    # 先找到一个checkpoint
    checkpoints = [f for f in os.listdir("log") if f.startswith("model_") and f.endswith(".pt")]
    if not checkpoints:
        print("没有找到checkpoint文件，跳过此测试")
        return False
    
    checkpoint_path = os.path.join("log", checkpoints[0])
    cmd = f"python model/train_gpt2.py --resume --checkpoint_path {checkpoint_path}"
    return run_command(cmd, "从指定Checkpoint恢复")

def test_auto_resume():
    """测试自动恢复功能"""
    cmd = "python model/train_gpt2.py --auto_resume"
    return run_command(cmd, "自动恢复功能")

def test_checkpoint_cleanup():
    """测试checkpoint清理功能"""
    cmd = "python model/train_gpt2.py --checkpoint_interval 2 --keep_last_n_checkpoints 2"
    success = run_command(cmd, "Checkpoint清理功能")
    
    if success:
        # 检查是否只保留了最新的2个checkpoint
        checkpoints = [f for f in os.listdir("log") if f.startswith("model_") and f.endswith(".pt")]
        print(f"保留的checkpoint文件: {checkpoints} (应该最多2个)")
        return len(checkpoints) <= 2
    return False

def main():
    parser = argparse.ArgumentParser(description="测试train_gpt2.py的所有功能")
    parser.add_argument("--no-modify", action="store_true", help="不修改训练脚本参数")
    parser.add_argument("--keep-logs", action="store_true", help="保留测试日志")
    args = parser.parse_args()
    
    print("开始测试train_gpt2.py的所有功能...")
    
    # 修改训练脚本以加快测试
    backup_path = None
    if not args.no_modify:
        backup_path = modify_train_script()
        print("已修改训练脚本参数以加快测试")
    
    # 清理测试环境
    if not args.keep_logs:
        clean_test_env()
    
    # 运行所有测试
    test_results = {
        "基本训练": False,
        "Checkpoint保存": False,
        "从Checkpoint恢复": False,
        "自动恢复": False,
        "Checkpoint清理": False,
    }
    
    try:
        # 测试1: 基本训练
        test_results["基本训练"] = test_basic_training()
        
        # 测试2: Checkpoint保存
        test_results["Checkpoint保存"] = test_checkpoint_save()
        
        # 测试3: 从指定checkpoint恢复
        test_results["从Checkpoint恢复"] = test_resume_from_checkpoint()
        
        # 测试4: 自动恢复
        test_results["自动恢复"] = test_auto_resume()
        
        # 测试5: Checkpoint清理
        test_results["Checkpoint清理"] = test_checkpoint_cleanup()
        
    finally:
        # 恢复原始训练脚本
        if backup_path and not args.no_modify:
            restore_train_script(backup_path)
            print("\n已恢复原始训练脚本")
    
    # 输出测试总结
    print("\n" + "="*60)
    print("测试总结:")
    print("="*60)
    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    # 显示log目录内容
    print("\nlog目录内容:")
    for file in sorted(os.listdir("log")):
        file_path = os.path.join("log", file)
        size = os.path.getsize(file_path) / 1024 / 1024  # MB
        print(f"  - {file} ({size:.2f} MB)")

if __name__ == "__main__":
    main()