"""
数据集完整性检查脚本
检查标注文件是否有格式错误

Usage:
    python check_dataset.py --dataset_root ./dataset/IRDST/IRDST_real/
"""

import os
import argparse


def check_dataset(dataset_root):
    """检查数据集完整性"""
    print("=" * 60)
    print("数据集完整性检查")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 检查目录结构
    train_img_dir = os.path.join(dataset_root, "train", "images")
    train_label_dir = os.path.join(dataset_root, "train", "labels")
    test_img_dir = os.path.join(dataset_root, "test", "images")
    test_label_dir = os.path.join(dataset_root, "test", "labels")
    
    for dir_path, dir_name in [
        (train_img_dir, "train/images"),
        (train_label_dir, "train/labels"),
        (test_img_dir, "test/images"),
        (test_label_dir, "test/labels"),
    ]:
        if not os.path.exists(dir_path):
            errors.append(f"目录不存在: {dir_name}")
        else:
            print(f"✓ {dir_name}: {len(os.listdir(dir_path))} 文件")
    
    # 检查标注文件格式
    print("\n" + "-" * 60)
    print("检查标注文件格式...")
    print("-" * 60)
    
    for split in ["train", "test"]:
        label_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.exists(label_dir):
            continue
        
        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".txt"):
                continue
            
            label_path = os.path.join(label_dir, label_file)
            
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 检查格式
                    if line.startswith("[") and line.endswith("]"):
                        line_content = line.strip("[]").replace(" ", ",")
                    else:
                        line_content = line.replace(" ", ",")
                    
                    values = [x.strip() for x in line_content.split(",") if x.strip()]
                    
                    # 检查是否都是数字
                    for i, v in enumerate(values):
                        try:
                            float(v)
                        except ValueError:
                            errors.append(
                                f"{split}/{label_file} 第{line_num}行: "
                                f"'{v}' 不是有效数字 (值{i+1})"
                            )
                    
                    # 检查值的数量
                    if len(values) != 4:
                        warnings.append(
                            f"{split}/{label_file} 第{line_num}行: "
                            f"有{len(values)}个值，期望4个"
                        )
                        
            except Exception as e:
                errors.append(f"读取文件失败: {split}/{label_file} - {e}")
    
    # 打印结果
    print("\n" + "=" * 60)
    print("检查结果")
    print("=" * 60)
    
    if errors:
        print(f"\n❌ 发现 {len(errors)} 个错误:")
        for err in errors[:20]:  # 只显示前20个
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... 还有 {len(errors) - 20} 个错误")
    
    if warnings:
        print(f"\n⚠️ 发现 {len(warnings)} 个警告:")
        for warn in warnings[:10]:
            print(f"  - {warn}")
        if len(warnings) > 10:
            print(f"  ... 还有 {len(warnings) - 10} 个警告")
    
    if not errors and not warnings:
        print("\n✅ 数据集完整性检查通过！")
    
    return errors, warnings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集完整性检查")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./dataset/IRDST/IRDST_real/",
        help="数据集根目录",
    )
    args = parser.parse_args()
    
    check_dataset(args.dataset_root)
