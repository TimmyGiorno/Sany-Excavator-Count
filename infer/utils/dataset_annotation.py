import os
import json
import cv2
import glob
import re
from pathlib import Path


class DatasetAnnotationTool:
    def __init__(self, image_dir, output_file):
        """
        Initialize annotation tool

        Args:
            image_dir: Directory containing post images
            output_file: Path to output annotation file (JSON format)
        """
        self.image_dir = Path(image_dir)
        self.output_file = output_file
        self.image_pairs = []
        self.annotations = []
        self.current_index = 0

        # Load existing annotations
        self.load_existing_annotations()

        # Get all post images and sort by sequence number
        self.image_paths = self.get_sorted_images()
        print(f"Found {len(self.image_paths)} images")

        # Generate all consecutive image pairs
        self.generate_image_pairs()
        print(f"Generated {len(self.image_pairs)} image pairs to annotate")

    def get_sorted_images(self):
        """Get all post images and sort by sequence number"""
        # Pattern: xxxxxx_post_x.jpg
        pattern = re.compile(r'(\d+)_post_\d+\.jpg')

        image_files = []
        for img_path in self.image_dir.glob("*_post_*.jpg"):
            match = pattern.search(img_path.name)
            if match:
                seq_num = int(match.group(1))
                image_files.append((seq_num, img_path))

        # Sort by sequence number
        image_files.sort(key=lambda x: x[0])
        return [img_path for _, img_path in image_files]

    def generate_image_pairs(self):
        """Generate consecutive image pairs"""
        for i in range(len(self.image_paths) - 1):
            pair = {
                'img1_path': str(self.image_paths[i]),
                'img2_path': str(self.image_paths[i + 1]),
                'pair_id': f"pair_{i + 1:04d}",
                'seq1_num': self.get_seq_number(self.image_paths[i]),
                'seq2_num': self.get_seq_number(self.image_paths[i + 1])
            }
            self.image_pairs.append(pair)

    def get_seq_number(self, img_path):
        """Extract sequence number from image path"""
        pattern = re.compile(r'(\d+)_post_\d+\.jpg')
        match = pattern.search(img_path.name)
        if match:
            return int(match.group(1))
        return 0

    def load_existing_annotations(self):
        """Load existing annotations"""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
            print(f"Loaded {len(self.annotations)} existing annotations")
            self.current_index = len(self.annotations)

    def save_annotations(self):
        """Save annotations"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(self.annotations)} annotations to {self.output_file}")

    def display_images(self, img1_path, img2_path):
        """Display two images side by side"""
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"Cannot read images: {img1_path} or {img2_path}")
            return None

        # Resize images to same height (e.g., 300 pixels)
        height = 300
        img1_resized = self.resize_to_height(img1, height)
        img2_resized = self.resize_to_height(img2, height)

        # Concatenate horizontally
        combined = np.hstack([img1_resized, img2_resized])

        # Add separator line
        h, w = combined.shape[:2]
        cv2.line(combined, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

        # Add text labels (English only to avoid encoding issues)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Image A", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "Image B", (w // 2 + 10, 30), font, 0.7, (0, 255, 0), 2)

        return combined

    def resize_to_height(self, img, target_height):
        """Resize image to target height while maintaining aspect ratio"""
        h, w = img.shape[:2]
        ratio = target_height / h
        target_width = int(w * ratio)
        return cv2.resize(img, (target_width, target_height))

    def annotate_pair(self, pair):
        """Annotate a single image pair"""
        img_combined = self.display_images(pair['img1_path'], pair['img2_path'])
        if img_combined is None:
            return None

        # Use English-only window title
        window_name = f"Annotate - {pair['pair_id']} (ID:{pair['seq1_num']}->{pair['seq2_num']})"
        cv2.imshow(window_name, img_combined)

        print(f"\nCurrent pair: {pair['pair_id']}")
        print(f"  Left image ID: {pair['seq1_num']}")
        print(f"  Right image ID: {pair['seq2_num']}")
        print("Press key to label:")
        print("  [1] Positive - Same truck")
        print("  [0] Negative - Different truck")
        print("  [s] Skip this pair")
        print("  [q] Save and quit")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('1'):
                label = 1
                label_text = "positive"
                break
            elif key == ord('0'):
                label = 0
                label_text = "negative"
                break
            elif key == ord('s'):
                label = None
                label_text = "skipped"
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return "quit"
            else:
                print("Invalid key, press 1, 0, s or q")

        cv2.destroyWindow(window_name)

        if label is not None:
            annotation = {
                'pair_id': pair['pair_id'],
                'img1_path': pair['img1_path'],
                'img2_path': pair['img2_path'],
                'seq1_num': pair['seq1_num'],
                'seq2_num': pair['seq2_num'],
                'label': label,
                'label_text': label_text
            }
            return annotation
        return None

    def run(self):
        """Run annotation tool"""
        print("\n" + "=" * 60)
        print("Training Set Annotation Tool")
        print("=" * 60)
        print(f"Image directory: {self.image_dir}")
        print(f"Output file: {self.output_file}")
        print(f"Pairs to annotate: {len(self.image_pairs) - self.current_index}")
        print("=" * 60 + "\n")

        for i in range(self.current_index, len(self.image_pairs)):
            pair = self.image_pairs[i]
            result = self.annotate_pair(pair)

            if result == "quit":
                break
            elif result is not None:
                self.annotations.append(result)
                self.current_index += 1
                # Save every 10 annotations
                if len(self.annotations) % 10 == 0:
                    self.save_annotations()
                    print(f"Auto-saved, progress: {self.current_index}/{len(self.image_pairs)}")

        # Final save
        self.save_annotations()

        # Statistics
        positive_count = sum(1 for ann in self.annotations if ann['label'] == 1)
        negative_count = sum(1 for ann in self.annotations if ann['label'] == 0)

        print("\n" + "=" * 60)
        print("Annotation Complete!")
        print(f"Total annotated: {len(self.annotations)} pairs")
        print(f"Positive (same truck): {positive_count} pairs")
        print(f"Negative (different truck): {negative_count} pairs")
        print(f"Output file: {self.output_file}")
        print("=" * 60)

        # Generate training config
        self.generate_train_config()

    def generate_train_config(self):
        """Generate training configuration"""
        config = {
            'dataset_info': {
                'total_pairs': len(self.annotations),
                'positive_pairs': sum(1 for ann in self.annotations if ann['label'] == 1),
                'negative_pairs': sum(1 for ann in self.annotations if ann['label'] == 0),
                'image_size': [320, 320],
                'description': 'Mining truck continuity annotation dataset'
            },
            'train_annotations': self.output_file,
            'image_directory': str(self.image_dir),
            'training_params': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 50,
                'input_size': 224,
                'pretrained': True
            }
        }

        config_file = self.output_file.replace('.json', '_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"\nGenerated training config file: {config_file}")


def main():
    # Configure paths
    IMAGE_DIR = r"E:\pycharmProjects\Sany-Excavator-Count\infer\cvnet_trainset"
    OUTPUT_FILE = r"E:\pycharmProjects\Sany-Excavator-Count\infer\cvnet_annotations.json"

    # Create annotation tool
    annotator = DatasetAnnotationTool(IMAGE_DIR, OUTPUT_FILE)

    # Run annotation
    annotator.run()


if __name__ == "__main__":
    import numpy as np

    main()