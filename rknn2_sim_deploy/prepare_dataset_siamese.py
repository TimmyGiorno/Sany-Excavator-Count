import os
import glob

def generate_dataset_file(folder_path, output_file):
    search_path = os.path.join(folder_path, "*.jpg")
    files = glob.glob(search_path)

    files.sort()

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for file_path in files:
                f.write(f"{file_path.replace(chr(92), '/')}\n")
        print(f"成功！已将 {len(files)} 个文件路径写入到 {output_file}")
    except Exception as e:
        print(f"发生错误: {e}")


folder = "./siamese_calibration_imgs"
output = "siamese_dataset.txt"

if __name__ == "__main__":
    generate_dataset_file(folder, output)
