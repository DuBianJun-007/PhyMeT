"""
Create a mini dataset for testing code functionality
"""
import os
import shutil

SOURCE_DIR = r"e:\workspace\MemISTD\dataset\IRDST\IRDST_real"
TARGET_DIR = r"e:\workspace\MemISTD\dataset\IRDST\IRDST_mini"

TRAIN_IMAGES = ["00000000", "00000001", "00000002"]
TEST_IMAGES = ["00000003", "00000004"]

def create_mini_dataset():
    dirs_to_create = [
        "images/train",
        "images/test",
        "boxes/train",
        "boxes/test",
        "masks/train",
        "masks/test",
    ]
    
    for dir_path in dirs_to_create:
        full_path = os.path.join(TARGET_DIR, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created: {full_path}")
    
    print("\nCopying training images...")
    for img_id in TRAIN_IMAGES:
        src_img = os.path.join(SOURCE_DIR, "images", "train", f"{img_id}.png")
        dst_img = os.path.join(TARGET_DIR, "images", "train", f"{img_id}.png")
        shutil.copy2(src_img, dst_img)
        print(f"  Copied: {img_id}.png")
        
        src_box = os.path.join(SOURCE_DIR, "boxes", "train", f"{img_id}.txt")
        dst_box = os.path.join(TARGET_DIR, "boxes", "train", f"{img_id}.txt")
        shutil.copy2(src_box, dst_box)
        print(f"  Copied: {img_id}.txt (boxes)")
        
        src_mask = os.path.join(SOURCE_DIR, "masks", "train", f"{img_id}.png")
        dst_mask = os.path.join(TARGET_DIR, "masks", "train", f"{img_id}.png")
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask)
            print(f"  Copied: {img_id}.png (mask)")
    
    print("\nCopying test images...")
    for img_id in TEST_IMAGES:
        src_img = os.path.join(SOURCE_DIR, "images", "test", f"{img_id}.png")
        dst_img = os.path.join(TARGET_DIR, "images", "test", f"{img_id}.png")
        shutil.copy2(src_img, dst_img)
        print(f"  Copied: {img_id}.png")
        
        src_box = os.path.join(SOURCE_DIR, "boxes", "test", f"{img_id}.txt")
        dst_box = os.path.join(TARGET_DIR, "boxes", "test", f"{img_id}.txt")
        shutil.copy2(src_box, dst_box)
        print(f"  Copied: {img_id}.txt (boxes)")
        
        src_mask = os.path.join(SOURCE_DIR, "masks", "test", f"{img_id}.png")
        dst_mask = os.path.join(TARGET_DIR, "masks", "test", f"{img_id}.png")
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask)
            print(f"  Copied: {img_id}.png (mask)")
    
    print("\n" + "=" * 50)
    print("Mini dataset created successfully!")
    print(f"Location: {TARGET_DIR}")
    print(f"Training images: {len(TRAIN_IMAGES)}")
    print(f"Test images: {len(TEST_IMAGES)}")
    print("=" * 50)

if __name__ == "__main__":
    create_mini_dataset()
