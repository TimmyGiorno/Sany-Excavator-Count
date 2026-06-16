import os
import shutil
from pathlib import Path


def process_labels(labels_dir):
    """
    处理标签文件，进行类别映射到5个类：
        原2,3,4 -> 新2 (truck)
        原5     -> 新3 (loading)
        原6     -> 新4 (dumping)
        原0,1   -> 保持不变
    """
    labels_path = Path(labels_dir)

    # 定义类别转换映射 (原类别 -> 新类别)
    class_mapping = {
        0: 0,  # bucket-empty
        1: 1,  # bucket-full
        2: 2,  # truck-empty -> truck
        3: 2,  # truck-full -> truck
        4: 2,  # truck (原) -> truck
        5: 3,  # loading
        6: 4,  # dumping
        7: 5,  # mine
    }

    total_modified_files = 0
    total_processed_files = 0

    for video_folder in labels_path.iterdir():
        if not video_folder.is_dir():
            continue

        train_labels_dir = video_folder / 'labels' / 'train'

        if not train_labels_dir.exists():
            print(f"跳过 {video_folder.name}: 未找到 labels/train 目录")
            continue

        print(f"处理 {video_folder.name}...")

        # 备份原始标签
        # backup_dir = train_labels_dir.parent / 'train_backup'
        # if not backup_dir.exists():
        #     print(f"  创建备份到 {backup_dir}")
        #     shutil.copytree(train_labels_dir, backup_dir)

        modified_count = 0
        file_count = 0

        for label_file in train_labels_dir.glob('frame_*.txt'):
            file_count += 1
            modified = False
            new_lines = []

            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])

                # 应用类别转换
                if class_id in class_mapping:
                    new_class_id = class_mapping[class_id]
                    if new_class_id != class_id:
                        modified = True
                    parts[0] = str(new_class_id)
                    new_lines.append(' '.join(parts))
                else:
                    # 未知类别，保留原样并警告
                    print(f"    警告: {label_file.name} 中发现未知类别 {class_id}，保持不变")
                    new_lines.append(line.strip())

            if modified:
                with open(label_file, 'w') as f:
                    for line in new_lines:
                        f.write(line + '\n')
                modified_count += 1

        print(f"  处理了 {file_count} 个文件，其中 {modified_count} 个被修改")
        total_modified_files += modified_count
        total_processed_files += file_count

    return total_processed_files, total_modified_files


def main():
    # 设置路径 - 注意这里要指向datasetV1.2（根据你之前提供的路径）
    dataset_root = r'E:\pycharmProjects\Sany-Excavator-Count\train\datasets\datasetV1.3'

    print("=" * 60)
    print("YOLOv8 标签类别映射脚本")
    print("=" * 60)
    print("\n类别转换规则:")
    print("  原0 (bucket-empty) -> 新0 (bucket-empty)")
    print("  原1 (bucket-full)  -> 新1 (bucket-full)")
    print("  原2 (truck-empty)  -> 新2 (truck)")
    print("  原3 (truck-full)   -> 新2 (truck)")
    print("  原4 (truck)        -> 新2 (truck)")
    print("  原5 (loading)      -> 新3 (loading)")
    print("  原6 (dumping)      -> 新4 (dumping)")
    print("  原7 (mine)         -> 新5 (mine)")

    print("\n" + "=" * 60)
    print("开始处理标签文件")
    print("=" * 60)

    total_files, modified_files = process_labels(dataset_root)

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"总共处理文件数: {total_files}")
    print(f"修改文件数: {modified_files}")
    print("\n最终类别映射 (与你的data.yaml一致):")
    print("  - 类别0: bucket-empty")
    print("  - 类别1: bucket-full")
    print("  - 类别2: truck (合并后)")
    print("  - 类别3: loading")
    print("  - 类别4: dumping")
    print("  - 类别5: mine")
    print(f"\n处理后的标签文件位于: {dataset_root}")


if __name__ == "__main__":
    main()