import json
import os
import shutil
from pathlib import Path

def convert_coco_to_yolo(dataset_root, output_dir):
   
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    
    # Define splits
    splits = ['train', 'valid', 'test']
    
    # Class mapping: Map category IDs to 0-indexed YOLO classes
    # COCO Category 1: SINGLE PHASE UNFUSED TAP OFF UNIT -> YOLO class 0
    # COCO Category 2: ZERO PHASE SEQUENCE CURRENT TRANSFORMER -> YOLO class 1
    class_mapping = {
        1: 0, # SINGLE PHASE UNFUSED TAP OFF UNIT
        2: 1  # ZERO PHASE SEQUENCE CURRENT TRANSFORMER
    }
    class_names = [
        "SINGLE PHASE UNFUSED TAP OFF UNIT",
        "ZERO PHASE SEQUENCE CURRENT TRANSFORMER"
    ]
    
    print(f"Starting conversion from {dataset_root} to {output_dir}...")
    
    for split in splits:
        split_dir = dataset_root / split
        ann_file = split_dir / "_annotations.coco.json"
        
        if not ann_file.exists():
            print(f"Warning: {ann_file} not found. Skipping {split}.")
            continue
            
        with open(ann_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
            
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Build image lookup: image_id -> image metadata
        images_dict = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image_id
        annotations_by_img = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_img:
                annotations_by_img[img_id] = []
            annotations_by_img[img_id].append(ann)
            
        converted_images = 0
        total_boxes = 0
        
        for img_id, img_info in images_dict.items():
            file_name = img_info['file_name']
            src_img_path = split_dir / file_name
            
            if not src_img_path.exists():
                print(f"Image not found on disk: {src_img_path}")
                continue
                
            dst_img_path = out_img_dir / file_name
            # Copy or link image
            shutil.copy2(src_img_path, dst_img_path)
            
            img_w = float(img_info['width'])
            img_h = float(img_info['height'])
            
            # YOLO label file
            txt_file_name = Path(file_name).stem + ".txt"
            dst_txt_path = out_lbl_dir / txt_file_name
            
            anns = annotations_by_img.get(img_id, [])
            lines = []
            
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in class_mapping:
                    continue
                yolo_cls = class_mapping[cat_id]
                
                # COCO bbox: [x_min, y_min, width, height]
                x_min, y_min, w, h = ann['bbox']
                
                # Convert to YOLO normalized [x_center, y_center, width, height]
                x_center = (x_min + w / 2.0) / img_w
                y_center = (y_min + h / 2.0) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # Clip values to [0, 1] to prevent out-of-bound errors
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))
                
                lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                total_boxes += 1
                
            with open(dst_txt_path, 'w', encoding='utf-8') as lf:
                lf.write("\n".join(lines))
                
            converted_images += 1
            
        print(f"Split [{split.upper()}]: {converted_images} images copied, {total_boxes} boxes formatted into {out_lbl_dir}")
        
    # Create data.yaml
    yaml_content = f"""# SLD Symbol Detection Dataset - YOLO11 & YOLO Format
path: {output_dir.as_posix()}
train: images/train
val: images/valid
test: images/test

names:
  0: SINGLE PHASE UNFUSED TAP OFF UNIT
  1: ZERO PHASE SEQUENCE CURRENT TRANSFORMER

nc: 2
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yf.write(yaml_content)
        
    print(f"\nCreated dataset configuration at: {yaml_path}")
    print("Conversion completed successfully!")

if __name__ == "__main__":
    dataset_root = r"d:\HOSHO\Model Testing\SLD ANNOTATION.v2-version-2.coco"
    output_dir = r"d:\HOSHO\Model Testing\SLD_YOLO_Dataset"
    convert_coco_to_yolo(dataset_root, output_dir)
