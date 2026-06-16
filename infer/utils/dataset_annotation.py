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

    def resize_to_width(self, img, target_width):
        """Resize image to target width while maintaining aspect ratio"""
        h, w = img.shape[:2]
        ratio = target_width / w
        target_height = int(h * ratio)
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

    def create_visualization_grid(self, annotations, cols=4, thumb_height=150):
        """
        Create a grid visualization of multiple annotation pairs

        Args:
            annotations: List of annotation dictionaries
            cols: Number of columns in the grid
            thumb_height: Height of each thumbnail in pixels

        Returns:
            Grid image and metadata about positions
        """
        rows = (len(annotations) + cols - 1) // cols
        grid_images = []
        positions_info = []

        for idx, ann in enumerate(annotations):
            # Load and combine the two images
            img1 = cv2.imread(ann['img1_path'])
            img2 = cv2.imread(ann['img2_path'])

            if img1 is None or img2 is None:
                continue

            # Resize to same height
            img1_resized = self.resize_to_height(img1, thumb_height)
            img2_resized = self.resize_to_height(img2, thumb_height)

            # Combine horizontally
            pair_img = np.hstack([img1_resized, img2_resized])

            # Add label indicator
            h, w = pair_img.shape[:2]
            label_color = (0, 255, 0) if ann['label'] == 1 else (0, 0, 255)
            label_text = f"{ann['label_text'].upper()}"

            # Draw label background
            label_bg = np.full((40, w, 3), label_color, dtype=np.uint8)
            pair_with_label = np.vstack([label_bg, pair_img])

            # Add text
            cv2.putText(pair_with_label, label_text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add pair ID
            cv2.putText(pair_with_label, ann['pair_id'], (w - 150, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add sequence numbers
            seq_text = f"{ann['seq1_num']}->{ann['seq2_num']}"
            cv2.putText(pair_with_label, seq_text, (10, h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            positions_info.append({
                'index': idx,
                'annotation': ann,
                'row': idx // cols,
                'col': idx % cols
            })

            grid_images.append(pair_with_label)

        # Create grid
        if not grid_images:
            return None, []

        # Ensure all images have the same size
        max_height = max(img.shape[0] for img in grid_images)
        max_width = max(img.shape[1] for img in grid_images)

        # Pad images to same size
        padded_images = []
        for img in grid_images:
            h, w = img.shape[:2]
            if h < max_height or w < max_width:
                padded = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                padded[:h, :w] = img
                padded_images.append(padded)
            else:
                padded_images.append(img)

        # Arrange in grid
        row_images = []
        for r in range(rows):
            start_idx = r * cols
            end_idx = min(start_idx + cols, len(padded_images))
            row_imgs = padded_images[start_idx:end_idx]

            # Pad row if needed
            while len(row_imgs) < cols:
                blank = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                row_imgs.append(blank)

            row_concat = np.hstack(row_imgs)
            row_images.append(row_concat)

        # Add separator lines between rows
        separator_height = 5
        final_grid = []
        for i, row_img in enumerate(row_images):
            if i > 0:
                separator = np.full((separator_height, row_img.shape[1], 3), 100, dtype=np.uint8)
                final_grid.append(separator)
            final_grid.append(row_img)

        grid_image = np.vstack(final_grid) if final_grid else None

        return grid_image, positions_info

    def review_annotations_visual(self, batch_size=40, grid_cols=4, thumb_height=120):
        """
        Review annotations with visual grid display

        Args:
            batch_size: Number of pairs to display per batch
            grid_cols: Number of columns in visualization grid
            thumb_height: Height of each thumbnail
        """
        if not self.annotations:
            print("No annotations to review!")
            return

        print("\n" + "=" * 80)
        print("VISUAL REVIEW MODE - Checking Annotations")
        print("=" * 80)
        print(f"Total annotations: {len(self.annotations)}")

        pos_count = sum(1 for ann in self.annotations if ann['label'] == 1)
        neg_count = sum(1 for ann in self.annotations if ann['label'] == 0)
        print(f"Positive (same truck): {pos_count} pairs  |  Negative (different): {neg_count} pairs")
        print("=" * 80)

        # Review in batches
        total_pairs = len(self.annotations)
        current_batch_start = 0

        while current_batch_start < total_pairs:
            batch_end = min(current_batch_start + batch_size, total_pairs)
            batch = self.annotations[current_batch_start:batch_end]

            # Create visualization grid
            print(f"\nGenerating visualization grid for pairs {current_batch_start + 1} to {batch_end}...")
            grid_image, positions_info = self.create_visualization_grid(batch, grid_cols, thumb_height)

            if grid_image is None:
                print("Failed to create grid visualization")
                break

            # Add batch info header
            h, w = grid_image.shape[:2]
            header_height = 60
            header = np.full((header_height, w, 3), 50, dtype=np.uint8)

            # Add title
            cv2.putText(header,
                        f"ANNOTATION REVIEW - Batch {current_batch_start // batch_size + 1}/{(total_pairs + batch_size - 1) // batch_size}",
                        (w // 2 - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(header, f"Pairs {current_batch_start + 1}-{batch_end} of {total_pairs}",
                        (w // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Add legend
            legend_x = w - 300
            cv2.rectangle(header, (legend_x, 10), (legend_x + 20, 30), (0, 255, 0), -1)
            cv2.putText(header, "Positive", (legend_x + 25, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(header, (legend_x + 100, 10), (legend_x + 120, 30), (0, 0, 255), -1)
            cv2.putText(header, "Negative", (legend_x + 125, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add instruction footer
            footer_height = 80
            footer = np.full((footer_height, w, 3), 30, dtype=np.uint8)
            instructions = [
                "Press key:",
                "  [1-9] - View specific pair (by grid position)",
                "  [c] - Correct a pair",
                "  [s] - Save annotations",
                "  [n] - Next batch",
                "  [p] - Previous batch",
                "  [q] - Quit review"
            ]

            for i, text in enumerate(instructions):
                cv2.putText(footer, text, (10, 20 + i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Combine everything
            display_img = np.vstack([header, grid_image, footer])

            # Show the grid
            window_name = f"Annotation Review - Batch {current_batch_start // batch_size + 1}"
            cv2.imshow(window_name, display_img)

            # Display grid position numbers for reference
            print("\n" + "-" * 80)
            print(f"Batch {current_batch_start // batch_size + 1} Summary:")
            print(f"{'Pos':<4} {'Pair ID':<12} {'Label':<10} {'Sequence':<15} {'Status'}")
            print("-" * 80)

            for i, ann in enumerate(batch):
                label_str = "POSITIVE" if ann['label'] == 1 else "NEGATIVE"
                seq_str = f"{ann['seq1_num']} -> {ann['seq2_num']}"
                pos_num = i + 1
                print(f"{pos_num:<4} {ann['pair_id']:<12} {label_str:<10} {seq_str:<15}")

            print("-" * 80)
            print(f"Grid layout: {grid_cols} columns, numbers 1-{len(batch)} correspond to grid positions")
            print("Press 'n' for next batch, 'p' for previous, 'q' to quit")

            # Handle user input
            while True:
                key = cv2.waitKey(0) & 0xFF

                # Number keys 1-9 to view specific pairs
                if ord('1') <= key <= ord('9'):
                    pair_num = key - ord('0')
                    if 1 <= pair_num <= len(batch):
                        self.display_single_pair_enhanced(batch[pair_num - 1],
                                                          current_batch_start + pair_num)
                        # Redisplay the grid after closing
                        cv2.imshow(window_name, display_img)
                    else:
                        print(f"Pair {pair_num} not in current batch (1-{len(batch)})")

                # Correction mode
                elif key == ord('c'):
                    print("\nEnter grid position number to correct (1-{}):".format(len(batch)))
                    pos_input = input().strip()
                    if pos_input.isdigit():
                        pos = int(pos_input)
                        if 1 <= pos <= len(batch):
                            self.correct_annotation_enhanced(batch[pos - 1],
                                                             current_batch_start + pos - 1)
                            # Reload annotations and refresh batch
                            self.load_existing_annotations()
                            batch = self.annotations[current_batch_start:batch_end]
                            # Regenerate grid
                            grid_image, _ = self.create_visualization_grid(batch, grid_cols, thumb_height)
                            if grid_image is not None:
                                display_img = np.vstack([header, grid_image, footer])
                                cv2.imshow(window_name, display_img)
                            # Update summary
                            pos_count = sum(1 for ann in self.annotations if ann['label'] == 1)
                            neg_count = sum(1 for ann in self.annotations if ann['label'] == 0)
                            print(f"\nUpdated totals - Positive: {pos_count}, Negative: {neg_count}")
                        else:
                            print(f"Invalid position. Please enter 1-{len(batch)}")
                    else:
                        print("Invalid input")

                # Save
                elif key == ord('s'):
                    self.save_annotations()
                    print("Annotations saved!")

                # Next batch
                elif key == ord('n'):
                    current_batch_start += batch_size
                    cv2.destroyWindow(window_name)
                    break

                # Previous batch
                elif key == ord('p'):
                    current_batch_start = max(0, current_batch_start - batch_size)
                    cv2.destroyWindow(window_name)
                    break

                # Quit
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    print("\nReview completed!")
                    return

                else:
                    print("Invalid key. Press n/p/c/s/q or number key (1-9)")

        cv2.destroyAllWindows()
        print("\n" + "=" * 80)
        print("Review completed!")
        print(f"Final statistics: Positive: {pos_count}, Negative: {neg_count}")
        print("=" * 80)

    def display_single_pair_enhanced(self, annotation, pair_number):
        """Display a single annotated pair with enhanced visualization"""
        img_combined = self.display_images(annotation['img1_path'], annotation['img2_path'])
        if img_combined is None:
            return

        # Add detailed information
        h, w = img_combined.shape[:2]

        # Create info panel
        info_height = 120
        info_panel = np.full((info_height, w, 3), 40, dtype=np.uint8)

        # Add annotation details
        label_color = (0, 255, 0) if annotation['label'] == 1 else (0, 0, 255)
        label_text = f"LABEL: {annotation['label_text'].upper()}"

        cv2.putText(info_panel, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)
        cv2.putText(info_panel, f"Pair ID: {annotation['pair_id']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(info_panel, f"Sequence: {annotation['seq1_num']} -> {annotation['seq2_num']}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(info_panel, f"Pair #{pair_number} of {len(self.annotations)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Add instruction panel
        instruction_height = 60
        instruction_panel = np.full((instruction_height, w, 3), 30, dtype=np.uint8)
        cv2.putText(instruction_panel, "Press [c] to correct | [ESC] to return to grid",
                    (w // 2 - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Combine all
        display_img = np.vstack([info_panel, img_combined, instruction_panel])

        window_name = f"Detailed View - Pair #{pair_number}"
        cv2.imshow(window_name, display_img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('c'):
                cv2.destroyWindow(window_name)
                self.correct_annotation_enhanced(annotation, pair_number - 1)
                break
            elif key == 27:  # ESC key
                cv2.destroyWindow(window_name)
                break

    def correct_annotation_enhanced(self, annotation, annotation_index):
        """Correct a specific annotation with visual feedback"""
        print(f"\n{'=' * 60}")
        print(f"CORRECTING ANNOTATION: {annotation['pair_id']}")
        print(f"{'=' * 60}")
        print(f"Current label: {annotation['label_text'].upper()}")

        # Display the pair
        self.display_single_pair_enhanced(annotation, annotation_index + 1)

        print("\nSelect new label:")
        print("  [1] Positive - Same truck")
        print("  [0] Negative - Different truck")
        print("  [k] Keep current label")
        print("  [ESC] Cancel")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('1'):
                annotation['label'] = 1
                annotation['label_text'] = 'positive'
                print("\n✓ Label updated to: POSITIVE")
                break
            elif key == ord('0'):
                annotation['label'] = 0
                annotation['label_text'] = 'negative'
                print("\n✓ Label updated to: NEGATIVE")
                break
            elif key == ord('k'):
                print("\n→ Label kept unchanged")
                break
            elif key == 27:  # ESC
                print("\n→ Correction cancelled")
                cv2.destroyAllWindows()
                return
            else:
                print("Invalid key, press 1, 0, k, or ESC")

        # Update the annotations list
        self.annotations[annotation_index] = annotation
        self.save_annotations()
        cv2.destroyAllWindows()

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

        # Ask if user wants to review annotations
        print("\n" + "=" * 60)
        print("Review Options:")
        print("  1. Visual grid review (recommended)")
        print("  2. Skip review")
        print("=" * 60)

        review_choice = input("Enter choice (1/2): ").strip()

        if review_choice == '1':
            # Ask for grid configuration
            try:
                cols = int(input("Number of columns in grid (default 4): ").strip() or "4")
                cols = max(1, min(6, cols))  # Limit between 1 and 6
            except:
                cols = 4

            try:
                thumb_height = int(input("Thumbnail height in pixels (default 120): ").strip() or "120")
                thumb_height = max(80, min(200, thumb_height))  # Limit between 80 and 200
            except:
                thumb_height = 120

            self.review_annotations_visual(batch_size=40, grid_cols=cols, thumb_height=thumb_height)
        else:
            print("Skipping review...")

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
    IMAGE_DIR = r"E:\pycharmProjects\Sany-Excavator-Count\infer\siamese_trainset_V1.1"
    OUTPUT_FILE = r"E:\pycharmProjects\Sany-Excavator-Count\infer\siamese_trainset_V1.1\cvnet_annotations.json"

    # Create annotation tool
    annotator = DatasetAnnotationTool(IMAGE_DIR, OUTPUT_FILE)

    # Run annotation
    annotator.run()


if __name__ == "__main__":
    import numpy as np

    main()