#!/usr/bin/env python3
"""
数据集划分脚本 - 按8:1:1比例划分完整数据集
适用于GeoHDS项目的数据格式
"""

import os
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import argparse


def create_directory_structure(base_dir):
    """
    创建数据集目录结构
    """
    dirs_to_create = [
        os.path.join(base_dir, 'train'),
        os.path.join(base_dir, 'valid'),
        os.path.join(base_dir, 'test')
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")


def split_dataset(input_csv_path, output_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    按8:1:1比例划分数据集

    Args:
        input_csv_path: 完整数据集CSV文件路径
        output_dir: 输出目录路径
        train_ratio: 训练集比例 (默认0.8)
        valid_ratio: 验证集比例 (默认0.1)
        test_ratio: 测试集比例 (默认0.1)
        random_seed: 随机种子 (默认42)
    """

    # 验证比例
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和必须为1.0，当前为: {train_ratio + valid_ratio + test_ratio}")

    print(f"开始划分数据集...")
    print(
        f"划分比例 - 训练集: {train_ratio * 100:.1f}%, 验证集: {valid_ratio * 100:.1f}%, 测试集: {test_ratio * 100:.1f}%")
    print(f"随机种子: {random_seed}")

    # 读取完整数据集
    print(f"读取数据文件: {input_csv_path}")
    full_df = pd.read_csv(input_csv_path)
    total_samples = len(full_df)
    print(f"总样本数: {total_samples}")

    # 检查必要的列
    required_columns = ['pdbid', '-logKd/Ki']
    for col in required_columns:
        if col not in full_df.columns:
            raise ValueError(f"CSV文件中缺少必要的列: {col}")

    # 显示亲和力值统计
    print(f"\n亲和力值 (-logKd/Ki) 统计:")
    print(full_df['-logKd/Ki'].describe())

    # 设置随机种子
    np.random.seed(random_seed)

    # 第一次划分：分出测试集 (10%)
    train_valid_df, test_df = train_test_split(
        full_df,
        test_size=test_ratio,
        random_state=random_seed,
        shuffle=True
    )

    # 第二次划分：从剩余90%中分出训练集和验证集
    # 验证集占总数据的10%，所以在剩余90%中占 10%/90% = 1/9
    valid_ratio_adjusted = valid_ratio / (train_ratio + valid_ratio)

    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_ratio_adjusted,
        random_state=random_seed,
        shuffle=True
    )

    # 显示划分结果
    print(f"\n数据集划分结果:")
    print(f"训练集: {len(train_df)} 样本 ({len(train_df) / total_samples * 100:.1f}%)")
    print(f"验证集: {len(valid_df)} 样本 ({len(valid_df) / total_samples * 100:.1f}%)")
    print(f"测试集: {len(test_df)} 样本 ({len(test_df) / total_samples * 100:.1f}%)")

    # 创建输出目录结构
    create_directory_structure(output_dir)

    # 保存CSV文件
    train_csv_path = os.path.join(output_dir, 'train.csv')
    valid_csv_path = os.path.join(output_dir, 'valid.csv')
    test_csv_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_csv_path, index=False)
    valid_df.to_csv(valid_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"\nCSV文件已保存:")
    print(f"训练集: {train_csv_path}")
    print(f"验证集: {valid_csv_path}")
    print(f"测试集: {test_csv_path}")

    # 显示每个集合的亲和力分布
    print(f"\n各集合亲和力分布:")
    for name, df in [('训练集', train_df), ('验证集', valid_df), ('测试集', test_df)]:
        print(f"\n{name}:")
        print(f"  均值: {df['-logKd/Ki'].mean():.3f}")
        print(f"  标准差: {df['-logKd/Ki'].std():.3f}")
        print(f"  范围: [{df['-logKd/Ki'].min():.3f}, {df['-logKd/Ki'].max():.3f}]")

    return train_df, valid_df, test_df


def copy_data_files(source_data_dir, target_data_dir, train_df, valid_df, test_df, graph_type="Graph_GeoHDS",
                    distance="5A"):
    """
    复制对应的.pyg数据文件到新的目录结构
    """

    print(f"\n开始复制数据文件...")
    print(f"源目录: {source_data_dir}")
    print(f"目标目录: {target_data_dir}")

    datasets = [
        ('train', train_df, os.path.join(target_data_dir, 'train')),
        ('valid', valid_df, os.path.join(target_data_dir, 'valid')),
        ('test', test_df, os.path.join(target_data_dir, 'test'))
    ]

    for dataset_name, df, target_dir in datasets:
        print(f"\n处理 {dataset_name} 集...")
        success_count = 0
        failed_count = 0

        for _, row in df.iterrows():
            # --- 【这里的修改是关键】 ---
            # 1. 从DataFrame读取原始值 (可能是 2340.0)
            pdbid_float = row['pdbid']
            # 2. 立即将其转换为整数形式的字符串 ("2340")
            pdbid = str(int(pdbid_float))
            # -------------------------

            # 现在，pdbid已经是字符串了，可以安全地用于构建路径
            source_compound_dir = os.path.join(source_data_dir, pdbid)
            source_file = os.path.join(source_compound_dir, f"{graph_type}-{pdbid}_{distance}.pyg")

            target_compound_dir = os.path.join(target_dir, pdbid)
            target_file = os.path.join(target_compound_dir, f"{graph_type}-{pdbid}_{distance}.pyg")

            if os.path.exists(source_file):
                os.makedirs(target_compound_dir, exist_ok=True)

                try:
                    shutil.copy2(source_file, target_file)
                    success_count += 1
                except Exception as e:
                    print(f"  复制失败 {pdbid}: {e}")
                    failed_count += 1
            else:
                print(f"  源文件不存在: {source_file}")
                failed_count += 1

        print(f"  {dataset_name}集处理完成: 成功 {success_count}, 失败 {failed_count}")

def main():
    parser = argparse.ArgumentParser(description='数据集划分脚本')
    parser.add_argument('--input_csv', type=str, required=True, help='完整数据集CSV文件路径')
    parser.add_argument('--source_data_dir', type=str, required=True, help='源数据目录路径 (包含所有.pyg文件)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例 (默认0.8)')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='验证集比例 (默认0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例 (默认0.1)')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子 (默认42)')
    parser.add_argument('--copy_files', action='store_true', help='是否复制数据文件')
    parser.add_argument('--graph_type', type=str, default='Graph_GeoHDS', help='图类型')
    parser.add_argument('--distance', type=str, default='5A', help='距离阈值')

    args = parser.parse_args()

    try:
        # 数据集划分
        train_df, valid_df, test_df = split_dataset(
            input_csv_path=args.input_csv,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )

        # 复制数据文件 (可选)
        if args.copy_files:
            copy_data_files(
                source_data_dir=args.source_data_dir,
                target_data_dir=args.output_dir,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                graph_type=args.graph_type,
                distance=args.distance
            )

        print(f"\n✅ 数据集划分完成!")
        print(f"输出目录: {args.output_dir}")

    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
