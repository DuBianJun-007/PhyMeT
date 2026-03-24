"""
MemISTD Dataset Loader for IRDST Dataset
=========================================

Dataset structure:
    dataset_root/
    ├── images/
    │   ├── train/
    │   └── test/
    ├── masks/
    │   ├── train/
    │   └── test/
    ├── boxes/
    │   ├── train/
    │   └── test/
    └── center/
        ├── train/
        └── test/

Based on the original IRDST preprocessing from datasetprecess.py
"""

import os
import random
import numpy as np
import torch
from PIL import Image, ImageOps
import math
from torch.utils.data.dataset import Dataset


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert("RGB")
        return image


def preprocess(image):
    image = image.astype(np.float32)
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def rand(a=0.0, b=1.0):
    return np.random.rand() * (b - a) + a


def random_horizontal_flip(image, p=0.5):
    if random.random() < p:
        image = np.flip(image, axis=1).copy()
    return image


def random_vertical_flip(image, p=0.5):
    if random.random() < p:
        image = np.flip(image, axis=0).copy()
    return image


def random_brightness(image, brightness=0.2):
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-brightness, brightness)
        image = np.clip(image * factor, 0, 1)
    return image


def random_contrast(image, contrast=0.2):
    if random.random() < 0.5:
        factor = random.uniform(1 - contrast, 1 + contrast)
        mean = image.mean()
        image = np.clip((image - mean) * factor + mean, 0, 1)
    return image


def random_gaussian_noise(image, noise=0.01):
    if random.random() < 0.5:
        noise_array = np.random.normal(0, noise, image.shape).astype(np.float32)
        image = np.clip(image + noise_array, 0, 1)
    return image


class IRDSTDataset(Dataset):
    """
    IRDST Dataset with original preprocessing from datasetprecess.py
    """

    def __init__(
        self,
        dataset_root="./dataset/IRDST/IRDST_real/",
        image_size=512,
        type="train",
        augment=True,
        crop_size=480,
        base_size=512,
        flip_augmentation=True,
        vflip_augmentation=False,
        brightness_jitter=0.2,
        contrast_jitter=0.2,
        gaussian_noise=0.01,
        scale_jitter=0.0,
        pad_only=False,
        pad_to_size=None,
        pad_divisor=32,
        pad_left_right=None,
    ):
        super().__init__()

        self.dataset_root = dataset_root
        self.image_size = image_size
        self.type = type
        self.augment = augment
        self.crop_size = crop_size
        self.base_size = base_size
        self.flip_augmentation = flip_augmentation
        self.vflip_augmentation = vflip_augmentation
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.gaussian_noise = gaussian_noise
        self.scale_jitter = scale_jitter
        self.pad_only = pad_only
        self.pad_to_size = pad_to_size
        self.pad_divisor = pad_divisor
        self.pad_left_right = pad_left_right

        self.images_dir = os.path.join(dataset_root, "images", type)
        self.masks_dir = os.path.join(dataset_root, "masks", type)
        self.boxes_dir = os.path.join(dataset_root, "boxes", type)

        self.file_names = sorted(
            [f for f in os.listdir(self.images_dir) if f.endswith(".png")]
        )
        self.length = len(self.file_names)

        print(f"IRDST Dataset loaded: {self.length} samples for {type}")

    def _pad_to_target(self, img, mask, boxes):
        # 确保 boxes 是 float32 类型
        if len(boxes) > 0:
            boxes = boxes.astype(np.float32)
        
        w, h = img.size

        if self.pad_to_size is not None:
            target_h, target_w = int(self.pad_to_size[0]), int(self.pad_to_size[1])
        else:
            target_h = int(math.ceil(h / self.pad_divisor) * self.pad_divisor)
            target_w = int(math.ceil(w / self.pad_divisor) * self.pad_divisor)

        if target_h < h or target_w < w:
            crop_x = random.randint(0, max(0, w - target_w))
            crop_y = random.randint(0, max(0, h - target_h))

            img = img.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            mask = mask.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))

            if len(boxes) > 0:
                boxes = boxes.copy().astype(np.float32)
                boxes[:, 0] -= crop_x
                boxes[:, 1] -= crop_y
                valid_mask = (
                    (boxes[:, 0] >= 0)
                    & (boxes[:, 1] >= 0)
                    & (boxes[:, 0] + boxes[:, 2] <= target_w)
                    & (boxes[:, 1] + boxes[:, 3] <= target_h)
                )
                boxes = boxes[valid_mask]

            return img, mask, boxes

        pad_h = target_h - h
        pad_w = target_w - w

        if self.pad_left_right is None:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            pad_left = int(self.pad_left_right[0])
            pad_right = int(self.pad_left_right[1])

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        if pad_left or pad_right or pad_top or pad_bottom:
            img = ImageOps.expand(
                img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0
            )
            mask = ImageOps.expand(
                mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0
            )
            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 0] += pad_left
                boxes[:, 1] += pad_top

        return img, mask, boxes

    def __len__(self):
        return self.length

    def _load_boxes(self, box_path):
        boxes = []
        if os.path.exists(box_path):
            with open(box_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        line_content = line.strip("[]").replace(" ", ",")
                    else:
                        line_content = line.replace(" ", ",")
                    
                    try:
                        values = [float(x) for x in line_content.split(",") if x.strip()]
                        if len(values) == 4:
                            x, y, w, h = values
                            boxes.append([x, y, w, h])
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Failed to parse box line: {line}")
                        continue
        return (
            np.array(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 4), dtype=np.float32)
        )

    def _sync_transform_train(self, img, mask, boxes):
        # 确保 boxes 是 float32 类型
        if len(boxes) > 0:
            boxes = boxes.astype(np.float32)
        
        if self.flip_augmentation and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes) > 0:
                boxes[:, 0] = img.width - boxes[:, 0] - boxes[:, 2]

        if self.vflip_augmentation and random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if len(boxes) > 0:
                boxes[:, 1] = img.height - boxes[:, 1] - boxes[:, 3]

        if self.scale_jitter and self.scale_jitter > 0:
            lo = max(1, int(self.base_size * (1.0 - float(self.scale_jitter))))
            hi = max(lo, int(self.base_size * (1.0 + float(self.scale_jitter))))
            long_size = random.randint(lo, hi)
        else:
            long_size = random.randint(
                int(self.base_size * 0.5), int(self.base_size * 2.0)
            )
        w, h = img.size

        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # boxes format: tlwh (x, y, w, h) in pixels
        if len(boxes) > 0:
            boxes[:, 0] = boxes[:, 0] * ow / w
            boxes[:, 2] = boxes[:, 2] * ow / w
            boxes[:, 1] = boxes[:, 1] * oh / h
            boxes[:, 3] = boxes[:, 3] * oh / h

        if ow < self.crop_size or oh < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            if len(boxes) > 0:
                # padding is (left=0, top=0, right=padw, bottom=padh)
                # tlwh boxes are unchanged since left/top pads are zero
                pass

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if len(boxes) > 0:
            # crop for tlwh: convert to xyxy, crop, then back to tlwh
            x1s = boxes[:, 0]
            y1s = boxes[:, 1]
            x2s = boxes[:, 0] + boxes[:, 2]
            y2s = boxes[:, 1] + boxes[:, 3]

            x1s = x1s - x1
            y1s = y1s - y1
            x2s = x2s - x1
            y2s = y2s - y1

            x1s = np.clip(x1s, 0, self.crop_size)
            y1s = np.clip(y1s, 0, self.crop_size)
            x2s = np.clip(x2s, 0, self.crop_size)
            y2s = np.clip(y2s, 0, self.crop_size)

            ws = x2s - x1s
            hs = y2s - y1s
            valid_mask = (ws > 1) & (hs > 1)

            if valid_mask.any():
                boxes = np.stack(
                    [x1s[valid_mask], y1s[valid_mask], ws[valid_mask], hs[valid_mask]],
                    axis=1,
                ).astype(np.float32)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)

        return img, mask, boxes

    def _testval_transform(self, img, mask, boxes):  # 测试数据集的处理，直接返回，表示原尺寸
        # 确保 boxes 是 float32 类型
        if len(boxes) > 0:
            boxes = boxes.astype(np.float32)
        # img, mask, boxes = self._pad_to_target(img, mask, boxes)
        return img, mask, boxes

    def _create_target_mask(self, boxes, width, height):
        target_mask = torch.zeros(1, height, width, dtype=torch.float32)

        if len(boxes) > 0:
            for box in boxes:
                try:
                    # 确保所有值都是数值类型
                    x = float(box[0])
                    y = float(box[1])
                    w = float(box[2])
                    h = float(box[3])
                    
                    x1 = int(max(0, x))
                    y1 = int(max(0, y))
                    x2 = int(min(width, x + w))
                    y2 = int(min(height, y + h))

                    if x2 > x1 and y2 > y1:
                        target_mask[0, y1:y2, x1:x2] = 1.0
                except (ValueError, TypeError, IndexError) as e:
                    # 跳过无效的 box
                    continue

        return target_mask

    def __getitem__(self, index):
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                actual_index = (index + retry_count) % len(self.file_names)
                file_name = self.file_names[actual_index]
                base_name = os.path.splitext(file_name)[0]

                img_path = os.path.join(self.images_dir, file_name)
                mask_path = os.path.join(self.masks_dir, file_name)
                box_path = os.path.join(self.boxes_dir, base_name + ".txt")

                if not os.path.exists(img_path):
                    print(f"WARNING: Image file not found: {img_path}")
                    retry_count += 1
                    continue

                if not os.path.exists(mask_path):
                    print(f"WARNING: Mask file not found: {mask_path}")
                    retry_count += 1
                    continue

                try:
                    img = Image.open(img_path).convert("L")
                except Exception as e:
                    print(f"WARNING: Failed to load image: {img_path}")
                    print(f"  Error: {str(e)}")
                    print(f"  Skipping to next sample...")
                    retry_count += 1
                    continue

                try:
                    mask = Image.open(mask_path).convert("1")
                except Exception as e:
                    print(f"WARNING: Failed to load mask: {mask_path}")
                    print(f"  Error: {str(e)}")
                    print(f"  Skipping to next sample...")
                    retry_count += 1
                    continue

                boxes = self._load_boxes(box_path)
                
                # 确保 boxes 是正确的数值类型
                if len(boxes) > 0:
                    boxes = boxes.astype(np.float32)

                if self.pad_only:
                    if (
                        self.type == "train"
                        and self.augment
                        and self.flip_augmentation
                        and random.random() < 0.5
                    ):
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                        if len(boxes) > 0:
                            boxes[:, 0] = img.width - boxes[:, 0] - boxes[:, 2]

                    if (
                        self.type == "train"
                        and self.augment
                        and self.vflip_augmentation
                        and random.random() < 0.5
                    ):
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                        if len(boxes) > 0:
                            boxes[:, 1] = img.height - boxes[:, 1] - boxes[:, 3]
                    img, mask, boxes = self._pad_to_target(img, mask, boxes)
                else:
                    if self.type == "train" and self.augment:
                        img, mask, boxes = self._sync_transform_train(img, mask, boxes)
                    else:
                        img, mask, boxes = self._testval_transform(img, mask, boxes)

                if self.type == "train" and self.augment:
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    if self.brightness_jitter and self.brightness_jitter > 0:
                        img_array = random_brightness(
                            img_array, brightness=float(self.brightness_jitter)
                        )
                    if self.contrast_jitter and self.contrast_jitter > 0:
                        img_array = random_contrast(
                            img_array, contrast=float(self.contrast_jitter)
                        )
                    if self.gaussian_noise and self.gaussian_noise > 0:
                        img_array = random_gaussian_noise(
                            img_array, noise=float(self.gaussian_noise)
                        )
                    img = Image.fromarray(np.clip(img_array * 255, 0, 255).astype(np.uint8))

                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = (img_array + 0.1246) / 1.0923

                img_tensor = torch.from_numpy(img_array).unsqueeze(0).type(torch.float32)
                mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).type(torch.float32)

                target_mask = self._create_target_mask(boxes, img.width, img.height)

                if len(boxes) > 0:
                    boxes_tensor = torch.from_numpy(boxes).type(torch.float32)
                else:
                    boxes_tensor = torch.zeros(0, 4, dtype=torch.float32)

                target = {
                    "boxes": boxes_tensor,
                    "labels": torch.zeros(boxes_tensor.size(0), dtype=torch.long)
                    if boxes_tensor.size(0) > 0
                    else torch.zeros(0, dtype=torch.long),
                    "image_id": torch.tensor(actual_index),
                    "target_mask": target_mask,
                    "mask": mask_tensor,
                    "file_name": file_name,
                }

                return img_tensor, target

            except Exception as e:
                print(f"WARNING: Unexpected error processing sample {index + retry_count}")
                print(f"  Error: {str(e)}")
                print(f"  Skipping to next sample...")
                retry_count += 1

        print(f"ERROR: Failed to load {max_retries} consecutive samples, returning dummy data")
        img_tensor = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        target = {
            "boxes": torch.zeros(0, 4, dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.long),
            "image_id": torch.tensor(index),
            "target_mask": torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32),
            "mask": torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32),
            "file_name": "dummy",
        }
        return img_tensor, target


def irdst_collate_fn(batch):
    images = []
    boxes_list = []
    labels_list = []
    masks_list = []
    target_masks_list = []
    image_ids = []
    file_names = []

    for img, target in batch:
        images.append(img)
        boxes_list.append(target["boxes"])
        labels_list.append(target["labels"])
        masks_list.append(target["mask"])
        target_masks_list.append(target["target_mask"])
        image_ids.append(target["image_id"])
        file_names.append(target["file_name"])

    images = torch.stack(images, dim=0)
    target_masks = torch.stack(target_masks_list, dim=0)

    targets = {
        "boxes": boxes_list,
        "labels": labels_list,
        "mask": masks_list,
        "target_mask": target_masks,
        "image_id": image_ids,
        "file_name": file_names,
        "num_boxes": torch.tensor([len(b) for b in boxes_list]),
    }

    return images, targets


class MemISTDDataset(Dataset):
    def __init__(
        self,
        dataset_path="./dataset/ITSDT-15K/",
        image_size=640,
        type="train",
        dataset_root="./dataset/ITSDT-15K/",
        augment=True,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.type = type
        self.dataset_root = dataset_root
        self.augment = augment

        self.img_idx = []
        self.anno_idx = []

        if os.path.exists(dataset_path):
            with open(dataset_path, encoding="utf-8") as f:
                data_lines = f.readlines()

            self.length = len(data_lines)

            for line in data_lines:
                line = line.strip("\n").split()

                if len(line) < 2:
                    continue

                img_path = line[0]

                if not os.path.isabs(img_path):
                    img_path = os.path.join(dataset_root, img_path)

                boxes = []
                for box_str in line[1:]:
                    try:
                        box_values = list(map(int, box_str.split(",")))
                        if len(box_values) == 4:
                            boxes.append(np.array(box_values, dtype=np.float32))
                    except (ValueError, IndexError):
                        continue

                if len(boxes) > 0:
                    self.img_idx.append(img_path)
                    self.anno_idx.append(np.array(boxes, dtype=np.float32))
        else:
            print(f"Warning: Dataset path does not exist: {dataset_path}")
            self.length = 0

        print(f"Loaded {self.length} samples for {type}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image, boxes = self.get_data(index)

        image = np.array(image, dtype=np.float32)
        image = preprocess(image)
        image = np.transpose(image, (2, 0, 1))

        # 确保 boxes 是 float32 类型
        if len(boxes) > 0:
            boxes = boxes.astype(np.float32)
            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            boxes[:, 0:2] = boxes[:, 0:2] + (boxes[:, 2:4] / 2)

        boxes = torch.from_numpy(boxes).type(torch.float32)

        target = {
            "boxes": boxes,
            "labels": torch.zeros(boxes.size(0), dtype=torch.long)
            if boxes.size(0) > 0
            else torch.zeros(0, dtype=torch.long),
            "image_id": torch.tensor(index),
        }

        if len(boxes) > 0 and self.type == "train":
            h, w = self.image_size, self.image_size
            target_mask = torch.zeros(1, h, w, dtype=torch.float32)

            for box in boxes:
                x1 = int(max(0, int(box[0] - box[2] / 2)))
                y1 = int(max(0, int(box[1] - box[3] / 2)))
                x2 = int(min(w, int(box[0] + box[2] / 2)))
                y2 = int(min(h, int(box[1] + box[3] / 2)))

                if x2 > x1 and y2 > y1:
                    target_mask[0, y1:y2, x1:x2] = 1.0

            target["target_mask"] = target_mask
        else:
            target_mask = torch.zeros(
                1, self.image_size, self.image_size, dtype=torch.float32
            )
            target["target_mask"] = target_mask

        image = torch.from_numpy(image).type(torch.float32)

        return image, target

    def get_data(self, index):
        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index]

        if os.path.exists(file_name):
            image = Image.open(file_name)
            image = cvtColor(image)
        else:
            image = Image.new("RGB", (w, h), (128, 128, 128))

        iw, ih = image.size

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_img = Image.new("RGB", (w, h), (128, 128, 128))
        new_img.paste(image, (dx, dy))

        label_data = self.anno_idx[index].copy()
        
        # 确保 label_data 是 float32 类型
        if len(label_data) > 0:
            label_data = np.array(label_data, dtype=np.float32)

        if len(label_data) > 0:
            if self.type == "train":
                np.random.shuffle(label_data)

            label_data[:, [0, 2]] = label_data[:, [0, 2]] * nw / iw + dx
            label_data[:, [1, 3]] = label_data[:, [1, 3]] * nh / ih + dy

            label_data[:, 0:2][label_data[:, 0:2] < 0] = 0
            label_data[:, 2][label_data[:, 2] > w] = w
            label_data[:, 3][label_data[:, 3] > h] = h

            box_w = label_data[:, 2] - label_data[:, 0]
            box_h = label_data[:, 3] - label_data[:, 1]

            label_data = label_data[np.logical_and(box_w > 1, box_h > 1)]

        image_array = np.array(new_img, dtype=np.float32)

        if self.augment and self.type == "train":
            image_array = random_horizontal_flip(image_array, p=0.5)
            image_array = random_vertical_flip(image_array, p=0.5)
            image_array = random_brightness(image_array, brightness=0.2)
            image_array = random_contrast(image_array, contrast=0.2)
            image_array = random_gaussian_noise(image_array, noise=0.01)

        label_data = np.array(label_data, dtype=np.float32)

        return Image.fromarray(np.uint8(image_array)), label_data


def dataset_collate(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets


# ============================================================
# ITSDTDataset：ITSDT-15K 多目标序列数据集（用于单帧检测）
# ============================================================

import xml.etree.ElementTree as ET


class ITSDTDataset(Dataset):
    """
    ITSDT-15K Dataset for single-frame infrared small target detection.

    Dataset structure:
        dataset_root/
        ├── Images/
        │   ├── 1/   (sequence 1)
        │   │   ├── 001.bmp
        │   │   └── ...
        │   └── ...
        └── Annotation/
            ├── 1/
            │   ├── 001.xml   (Pascal VOC format)
            │   └── ...
            └── ...

    Notes:
        - <name> field in XML is the track ID, ignored for detection (all → label 0)
        - Binary mask is generated from bounding boxes (no separate mask files)
        - Default split: seq 1~76 → train, seq 77~87 → test
    """

    def __init__(
        self,
        dataset_root="./dataset/ITSDT-15K/",
        image_size=480,
        type="train",
        augment=True,
        pad_only=True,
        pad_to_size=None,
        pad_divisor=4,
        flip_augmentation=True,
        vflip_augmentation=False,
        brightness_jitter=0.2,
        contrast_jitter=0.2,
        gaussian_noise=0.01,
        train_seq_range=(1, 76),
        test_seq_range=(77, 87),
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.type = type
        self.augment = augment
        self.pad_only = pad_only
        self.pad_to_size = pad_to_size
        self.pad_divisor = pad_divisor
        self.flip_augmentation = flip_augmentation
        self.vflip_augmentation = vflip_augmentation
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.gaussian_noise = gaussian_noise

        self.images_dir = os.path.join(dataset_root, "Images")
        self.anno_dir = os.path.join(dataset_root, "Annotation")

        # 根据 split 确定序列范围
        if type == "train":
            start_seq, end_seq = int(train_seq_range[0]), int(train_seq_range[1])
        else:
            start_seq, end_seq = int(test_seq_range[0]), int(test_seq_range[1])

        # 构建样本索引：(img_path, xml_path)
        self.samples = []
        for seq_id in range(start_seq, end_seq + 1):
            anno_seq_dir = os.path.join(self.anno_dir, str(seq_id))
            img_seq_dir = os.path.join(self.images_dir, str(seq_id))
            if not os.path.isdir(anno_seq_dir) or not os.path.isdir(img_seq_dir):
                continue
            xml_files = sorted(
                f for f in os.listdir(anno_seq_dir) if f.endswith(".xml")
            )
            for xml_name in xml_files:
                base = os.path.splitext(xml_name)[0]
                img_path = os.path.join(img_seq_dir, base + ".bmp")
                xml_path = os.path.join(anno_seq_dir, xml_name)
                if os.path.exists(img_path) and os.path.exists(xml_path):
                    self.samples.append((img_path, xml_path))

        self.length = len(self.samples)
        print(f"ITSDTDataset loaded: {self.length} samples for {type} "\
              f"(seq {start_seq}~{end_seq})")

    def __len__(self):
        return self.length

    def _parse_xml(self, xml_path):
        """解析 Pascal VOC XML，返回 bboxes: np.ndarray [N,4] (x,y,w,h 像素)。"""
        boxes = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                w = xmax - xmin
                h = ymax - ymin
                if w > 0 and h > 0:
                    boxes.append([xmin, ymin, w, h])
        except Exception as e:
            print(f"Warning: Failed to parse XML {xml_path}: {e}")
        return (
            np.array(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 4), dtype=np.float32)
        )

    def _pad_to_target(self, img, mask, boxes):
        """复用 IRDSTDataset 的 pad 逻辑（完全一致）。"""
        if len(boxes) > 0:
            boxes = boxes.astype(np.float32)
        w, h = img.size

        if self.pad_to_size is not None:
            target_h = int(self.pad_to_size[0])
            target_w = int(self.pad_to_size[1])
        else:
            target_h = int(math.ceil(h / self.pad_divisor) * self.pad_divisor)
            target_w = int(math.ceil(w / self.pad_divisor) * self.pad_divisor)

        if target_h < h or target_w < w:
            crop_x = random.randint(0, max(0, w - target_w))
            crop_y = random.randint(0, max(0, h - target_h))
            img = img.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            mask = mask.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            if len(boxes) > 0:
                boxes = boxes.copy().astype(np.float32)
                boxes[:, 0] -= crop_x
                boxes[:, 1] -= crop_y
                valid_mask = (
                    (boxes[:, 0] >= 0)
                    & (boxes[:, 1] >= 0)
                    & (boxes[:, 0] + boxes[:, 2] <= target_w)
                    & (boxes[:, 1] + boxes[:, 3] <= target_h)
                )
                boxes = boxes[valid_mask]
            return img, mask, boxes

        pad_h = target_h - h
        pad_w = target_w - w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        if pad_left or pad_right or pad_top or pad_bottom:
            img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            mask = ImageOps.expand(mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 0] += pad_left
                boxes[:, 1] += pad_top
        return img, mask, boxes

    def _create_target_mask(self, boxes, width, height):
        """从 bbox 生成二值 target mask。"""
        target_mask = torch.zeros(1, height, width, dtype=torch.float32)
        for box in boxes:
            try:
                x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                x1 = int(max(0, x))
                y1 = int(max(0, y))
                x2 = int(min(width,  x + w))
                y2 = int(min(height, y + h))
                if x2 > x1 and y2 > y1:
                    target_mask[0, y1:y2, x1:x2] = 1.0
            except (ValueError, TypeError):
                continue
        return target_mask

    def __getitem__(self, index):
        max_retries = 10
        for retry in range(max_retries):
            actual_index = (index + retry) % self.length
            img_path, xml_path = self.samples[actual_index]
            try:
                img = Image.open(img_path).convert("L")
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                continue

            boxes = self._parse_xml(xml_path)

            # 生成空 mask（后续由 bbox 填充）
            mask = Image.fromarray(np.zeros((img.height, img.width), dtype=np.uint8))

            # ---- 数据增强 ----
            if self.pad_only:
                if self.type == "train" and self.augment:
                    if self.flip_augmentation and random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                        if len(boxes) > 0:
                            boxes = boxes.copy()
                            boxes[:, 0] = img.width - boxes[:, 0] - boxes[:, 2]
                    if self.vflip_augmentation and random.random() < 0.5:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                        if len(boxes) > 0:
                            boxes = boxes.copy()
                            boxes[:, 1] = img.height - boxes[:, 1] - boxes[:, 3]
                img, mask, boxes = self._pad_to_target(img, mask, boxes)
            # 若 pad_only=False，直接使用原图（保持原始分辨率）

            # ---- 像素增强 ----
            if self.type == "train" and self.augment:
                img_array = np.array(img, dtype=np.float32) / 255.0
                if self.brightness_jitter and self.brightness_jitter > 0:
                    img_array = random_brightness(img_array, brightness=float(self.brightness_jitter))
                if self.contrast_jitter and self.contrast_jitter > 0:
                    img_array = random_contrast(img_array, contrast=float(self.contrast_jitter))
                if self.gaussian_noise and self.gaussian_noise > 0:
                    img_array = random_gaussian_noise(img_array, noise=float(self.gaussian_noise))
                img = Image.fromarray(np.clip(img_array * 255, 0, 255).astype(np.uint8))

            # ---- 归一化（ImageNet 灰度通道参数）----
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = (img_array - 0.485) / 0.229
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()

            # ---- mask tensor（全零，因为没有独立 mask 文件）----
            mask_tensor = torch.zeros(1, img.height, img.width, dtype=torch.float32)

            # ---- target mask（从 bbox 生成）----
            target_mask = self._create_target_mask(boxes, img.width, img.height)

            # ---- boxes tensor ----
            if len(boxes) > 0:
                boxes_tensor = torch.from_numpy(boxes.astype(np.float32))
            else:
                boxes_tensor = torch.zeros(0, 4, dtype=torch.float32)

            file_name = os.path.basename(img_path)
            target = {
                "boxes":       boxes_tensor,
                "labels":      torch.zeros(boxes_tensor.size(0), dtype=torch.long),
                "image_id":    torch.tensor(actual_index),
                "target_mask": target_mask,
                "mask":        mask_tensor,
                "file_name":   file_name,
            }
            return img_tensor, target

        # 所有重试失败，返回 dummy
        print(f"ERROR: Failed to load sample {index}, returning dummy data")
        img_tensor = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        target = {
            "boxes":       torch.zeros(0, 4, dtype=torch.float32),
            "labels":      torch.zeros(0, dtype=torch.long),
            "image_id":    torch.tensor(index),
            "target_mask": torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32),
            "mask":        torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32),
            "file_name":   "dummy",
        }
        return img_tensor, target


if __name__ == "__main__":
    dataset = IRDSTDataset(
        dataset_root="./dataset/IRDST/IRDST_real/",
        image_size=512,
        type="train",
        augment=False,
    )

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        img, target = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Boxes: {target['boxes'].shape}")


# =============================================================================
# ITSDT-15K Dataset
# =============================================================================

import xml.etree.ElementTree as ET


class ITSDTDataset(Dataset):
    """
    ITSDT-15K 数据集加载器（用于单帧红外小目标检测）。

    目录结构:
        dataset_root/
        ├── Images/<seq_id>/<frame>.bmp
        └── Annotation/<seq_id>/<frame>.xml  (Pascal VOC, xyxy)

    官方划分: 训练集序列 1-76，验证集序列 77-87。
    输出 boxes 格式: tlwh (x, y, w, h)，与 IRDSTDataset 一致。
    """

    TRAIN_SEQ_IDS = list(range(1, 77))
    TEST_SEQ_IDS  = list(range(77, 88))

    def __init__(
        self,
        dataset_root="./dataset/ITSDT-15K/",
        image_size=512,
        type="train",
        augment=True,
        pad_only=True,
        pad_to_size=None,
        pad_divisor=32,
        flip_augmentation=True,
        vflip_augmentation=False,
        brightness_jitter=0.2,
        contrast_jitter=0.2,
        gaussian_noise=0.01,
        crop_size=480,
        base_size=512,
        scale_jitter=0.0,
        pad_left_right=None,
    ):
        super().__init__()
        self.dataset_root        = dataset_root
        self.image_size          = image_size
        self.type                = type
        self.augment             = augment
        self.pad_only            = pad_only
        self.pad_to_size         = pad_to_size
        self.pad_divisor         = pad_divisor
        self.flip_augmentation   = flip_augmentation
        self.vflip_augmentation  = vflip_augmentation
        self.brightness_jitter   = brightness_jitter
        self.contrast_jitter     = contrast_jitter
        self.gaussian_noise      = gaussian_noise
        self.crop_size           = crop_size
        self.base_size           = base_size
        self.scale_jitter        = scale_jitter
        self.pad_left_right      = pad_left_right
        self.images_dir = os.path.join(dataset_root, "Images")
        self.anno_dir   = os.path.join(dataset_root, "Annotation")

        seq_ids = self.TRAIN_SEQ_IDS if type == "train" else self.TEST_SEQ_IDS
        self.samples = []
        for sid in seq_ids:
            sid_str      = str(sid)
            img_seq_dir  = os.path.join(self.images_dir, sid_str)
            anno_seq_dir = os.path.join(self.anno_dir,   sid_str)
            if not os.path.isdir(img_seq_dir):
                continue
            for fname in sorted(os.listdir(img_seq_dir)):
                if not fname.lower().endswith(".bmp"):
                    continue
                stem     = os.path.splitext(fname)[0]
                xml_path = os.path.join(anno_seq_dir, stem + ".xml")
                if os.path.exists(xml_path):
                    self.samples.append((sid_str, stem))

        self.length = len(self.samples)
        print(f"ITSDTDataset loaded: {self.length} samples for {type} "
              f"(sequences {seq_ids[0]}-{seq_ids[-1]})")

    def __len__(self):
        return self.length

    def _parse_xml(self, xml_path):
        """解析 VOC XML，返回 tlwh boxes: np.ndarray (N,4) float32。"""
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            print(f"WARNING: Failed to parse XML: {xml_path}")
            return np.zeros((0, 4), dtype=np.float32)
        boxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                xmin = float(bb.find("xmin").text)
                ymin = float(bb.find("ymin").text)
                xmax = float(bb.find("xmax").text)
                ymax = float(bb.find("ymax").text)
                w, h = xmax - xmin, ymax - ymin
                if w > 0 and h > 0:
                    boxes.append([xmin, ymin, w, h])
            except (AttributeError, ValueError, TypeError):
                continue
        return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

    def _pad_to_target(self, img, mask, boxes):
        if len(boxes) > 0:
            boxes = boxes.astype(np.float32)
        w, h = img.size
        if self.pad_to_size is not None:
            target_h, target_w = int(self.pad_to_size[0]), int(self.pad_to_size[1])
        else:
            target_h = int(math.ceil(h / self.pad_divisor) * self.pad_divisor)
            target_w = int(math.ceil(w / self.pad_divisor) * self.pad_divisor)
        if target_h < h or target_w < w:
            crop_x = random.randint(0, max(0, w - target_w))
            crop_y = random.randint(0, max(0, h - target_h))
            img  = img.crop( (crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            mask = mask.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            if len(boxes) > 0:
                boxes = boxes.copy().astype(np.float32)
                boxes[:, 0] -= crop_x
                boxes[:, 1] -= crop_y
                valid = (
                    (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) &
                    (boxes[:, 0] + boxes[:, 2] <= target_w) &
                    (boxes[:, 1] + boxes[:, 3] <= target_h)
                )
                boxes = boxes[valid]
            return img, mask, boxes
        pad_h = target_h - h
        pad_w = target_w - w
        if self.pad_left_right is None:
            pad_left  = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            pad_left  = int(self.pad_left_right[0])
            pad_right = int(self.pad_left_right[1])
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        if pad_left or pad_right or pad_top or pad_bottom:
            img  = ImageOps.expand(img,  border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            mask = ImageOps.expand(mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 0] += pad_left
                boxes[:, 1] += pad_top
        return img, mask, boxes

    def _create_target_mask(self, boxes, width, height):
        target_mask = torch.zeros(1, height, width, dtype=torch.float32)
        for box in boxes:
            try:
                x1 = int(max(0, float(box[0])))
                y1 = int(max(0, float(box[1])))
                x2 = int(min(width,  float(box[0]) + float(box[2])))
                y2 = int(min(height, float(box[1]) + float(box[3])))
                if x2 > x1 and y2 > y1:
                    target_mask[0, y1:y2, x1:x2] = 1.0
            except (ValueError, TypeError, IndexError):
                continue
        return target_mask

    def __getitem__(self, index):
        max_retries = 10
        for retry in range(max_retries):
            actual_index = (index + retry) % self.length
            sid_str, stem = self.samples[actual_index]
            img_path  = os.path.join(self.images_dir, sid_str, stem + ".bmp")
            xml_path  = os.path.join(self.anno_dir,   sid_str, stem + ".xml")
            file_name = f"{sid_str}/{stem}.bmp"
            try:
                img = Image.open(img_path).convert("L")
            except Exception as e:
                print(f"WARNING: Failed to load image {img_path}: {e}")
                continue
            boxes = self._parse_xml(xml_path)
            # ITSDT 无像素级 mask，用全零图代替
            mask = Image.fromarray(
                np.zeros((img.height, img.width), dtype=np.uint8)
            ).convert("1")
            # ---------- 数据增强 / padding ----------
            if self.pad_only:
                if self.type == "train" and self.augment:
                    if self.flip_augmentation and random.random() < 0.5:
                        img  = img.transpose(Image.FLIP_LEFT_RIGHT)
                        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                        if len(boxes) > 0:
                            boxes[:, 0] = img.width - boxes[:, 0] - boxes[:, 2]
                    if self.vflip_augmentation and random.random() < 0.5:
                        img  = img.transpose(Image.FLIP_TOP_BOTTOM)
                        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                        if len(boxes) > 0:
                            boxes[:, 1] = img.height - boxes[:, 1] - boxes[:, 3]
                img, mask, boxes = self._pad_to_target(img, mask, boxes)
            # ---------- 颜色增强（仅训练） ----------
            if self.type == "train" and self.augment:
                img_arr = np.array(img, dtype=np.float32) / 255.0
                if self.brightness_jitter > 0:
                    img_arr = random_brightness(img_arr, self.brightness_jitter)
                if self.contrast_jitter > 0:
                    img_arr = random_contrast(img_arr, self.contrast_jitter)
                if self.gaussian_noise > 0:
                    img_arr = random_gaussian_noise(img_arr, self.gaussian_noise)
                img = Image.fromarray(np.clip(img_arr * 255, 0, 255).astype(np.uint8))
            # ---------- 归一化（ImageNet 灰度通道参数） ----------
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_arr = (img_arr - 0.485) / 0.229
            img_tensor  = torch.from_numpy(img_arr).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(
                np.array(mask, dtype=np.float32)
            ).unsqueeze(0).float()
            target_mask = self._create_target_mask(
                boxes, img_tensor.shape[2], img_tensor.shape[1]
            )
            boxes_tensor = (
                torch.from_numpy(boxes).float()
                if len(boxes) > 0
                else torch.zeros(0, 4, dtype=torch.float32)
            )
            target = {
                "boxes":       boxes_tensor,
                "labels":      torch.zeros(boxes_tensor.size(0), dtype=torch.long),
                "image_id":    torch.tensor(actual_index),
                "target_mask": target_mask,
                "mask":        mask_tensor,
                "file_name":   file_name,
            }
            return img_tensor, target
        # 所有重试均失败，返回 dummy
        print(f"ERROR: ITSDTDataset failed to load sample {index}, returning dummy")
        dummy_h = self.pad_to_size[0] if self.pad_to_size else self.image_size
        dummy_w = self.pad_to_size[1] if self.pad_to_size else self.image_size
        img_tensor = torch.zeros(1, dummy_h, dummy_w, dtype=torch.float32)
        target = {
            "boxes":       torch.zeros(0, 4, dtype=torch.float32),
            "labels":      torch.zeros(0, dtype=torch.long),
            "image_id":    torch.tensor(index),
            "target_mask": torch.zeros(1, dummy_h, dummy_w, dtype=torch.float32),
            "mask":        torch.zeros(1, dummy_h, dummy_w, dtype=torch.float32),
            "file_name":   "dummy",
        }
        return img_tensor, target


# =============================================================================
# NUDT-MIRSDT Dataset
# 数据集结构:
#   dataset_root/
#   ├── SequenceX/
#   │   ├── images/  (XXXXX.png, 灰度)
#   │   └── masks/   (XXXXX.png, 0/255 二值)
#   ├── train.txt    (SequenceX/Mix/XXXXX.mat，每行一个样本)
#   └── test.txt
# =============================================================================

class NUDTMIRSDTDataset(Dataset):
    """
    NUDT-MIRSDT 数据集加载器（单帧红外小目标检测）。

    train.txt / test.txt 每行格式: SequenceX/Mix/XXXXX.mat
    图像路径:  dataset_root/SequenceX/images/XXXXX.png
    Mask 路径: dataset_root/SequenceX/masks/XXXXX.png  (0 / 255)

    Boxes 由 mask 连通域分析自动提取，格式为 tlwh (x, y, w, h)。
    """

    def __init__(
        self,
        dataset_root="./dataset/NUDT-MIRSDT/",
        image_size=256,
        type="train",
        augment=True,
        pad_only=True,
        pad_to_size=None,
        pad_divisor=4,
        flip_augmentation=True,
        vflip_augmentation=True,
        brightness_jitter=0.2,
        contrast_jitter=0.2,
        gaussian_noise=0.01,
        scale_jitter=0.0,
        crop_size=256,
        base_size=256,
        pad_left_right=None,
    ):
        super().__init__()
        self.dataset_root       = dataset_root
        self.image_size         = image_size
        self.type               = type
        self.augment            = augment
        self.pad_only           = pad_only
        self.pad_to_size        = pad_to_size
        self.pad_divisor        = pad_divisor
        self.flip_augmentation  = flip_augmentation
        self.vflip_augmentation = vflip_augmentation
        self.brightness_jitter  = brightness_jitter
        self.contrast_jitter    = contrast_jitter
        self.gaussian_noise     = gaussian_noise
        self.scale_jitter       = scale_jitter
        self.crop_size          = crop_size
        self.base_size          = base_size
        self.pad_left_right     = pad_left_right

        # 读取 split txt，把 .mat 路径转成 (img_path, mask_path)
        split_file = os.path.join(dataset_root, f"{type}.txt")
        self.samples = []
        if os.path.exists(split_file):
            with open(split_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # SequenceX/Mix/XXXXX.mat  ->  SequenceX, XXXXX
                    parts = line.replace("\\", "/").split("/")
                    if len(parts) < 3:
                        continue
                    seq_name = parts[0]                        # e.g. "Sequence19"
                    stem     = os.path.splitext(parts[-1])[0]  # e.g. "00001"
                    img_path  = os.path.join(dataset_root, seq_name, "images", stem + ".png")
                    mask_path = os.path.join(dataset_root, seq_name, "masks",  stem + ".png")
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))
        else:
            print(f"Warning: split file not found: {split_file}")

        self.length = len(self.samples)
        print(f"NUDTMIRSDTDataset loaded: {self.length} samples for {type}")

    def __len__(self):
        return self.length

    @staticmethod
    def _boxes_from_mask(mask_np):
        """
        从二值 mask (H,W uint8, 0/255) 提取连通域 bbox，返回 tlwh ndarray (N,4) float32。
        纯 NumPy + BFS 实现，无需 OpenCV。
        """
        from collections import deque
        binary = (mask_np > 128).astype(np.uint8)
        if binary.sum() == 0:
            return np.zeros((0, 4), dtype=np.float32)

        labeled  = np.zeros_like(binary, dtype=np.int32)
        label_id = 0
        rows, cols = binary.shape
        for r in range(rows):
            for c in range(cols):
                if binary[r, c] == 1 and labeled[r, c] == 0:
                    label_id += 1
                    queue = deque([(r, c)])
                    labeled[r, c] = label_id
                    while queue:
                        cr, cc = queue.popleft()
                        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols \
                               and binary[nr, nc] == 1 and labeled[nr, nc] == 0:
                                labeled[nr, nc] = label_id
                                queue.append((nr, nc))

        boxes = []
        for lid in range(1, label_id + 1):
            ys, xs = np.where(labeled == lid)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            w, h   = x2 - x1 + 1, y2 - y1 + 1
            if w > 0 and h > 0:
                boxes.append([float(x1), float(y1), float(w), float(h)])
        return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

    def _pad_to_target(self, img, mask, boxes):
        if len(boxes) > 0:
            boxes = boxes.astype(np.float32)
        w, h = img.size
        if self.pad_to_size is not None:
            target_h, target_w = int(self.pad_to_size[0]), int(self.pad_to_size[1])
        else:
            target_h = int(math.ceil(h / self.pad_divisor) * self.pad_divisor)
            target_w = int(math.ceil(w / self.pad_divisor) * self.pad_divisor)

        if target_h < h or target_w < w:
            crop_x = random.randint(0, max(0, w - target_w))
            crop_y = random.randint(0, max(0, h - target_h))
            img  = img.crop( (crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            mask = mask.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            if len(boxes) > 0:
                boxes = boxes.copy().astype(np.float32)
                boxes[:, 0] -= crop_x
                boxes[:, 1] -= crop_y
                valid = (
                    (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) &
                    (boxes[:, 0] + boxes[:, 2] <= target_w) &
                    (boxes[:, 1] + boxes[:, 3] <= target_h)
                )
                boxes = boxes[valid]
            return img, mask, boxes

        pad_h = target_h - h
        pad_w = target_w - w
        if self.pad_left_right is None:
            pad_left  = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            pad_left  = int(self.pad_left_right[0])
            pad_right = int(self.pad_left_right[1])
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        if pad_left or pad_right or pad_top or pad_bottom:
            img  = ImageOps.expand(img,  border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            mask = ImageOps.expand(mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 0] += pad_left
                boxes[:, 1] += pad_top
        return img, mask, boxes

    def _create_target_mask(self, boxes, width, height):
        target_mask = torch.zeros(1, height, width, dtype=torch.float32)
        for box in boxes:
            try:
                x1 = int(max(0, float(box[0])))
                y1 = int(max(0, float(box[1])))
                x2 = int(min(width,  float(box[0]) + float(box[2])))
                y2 = int(min(height, float(box[1]) + float(box[3])))
                if x2 > x1 and y2 > y1:
                    target_mask[0, y1:y2, x1:x2] = 1.0
            except (ValueError, TypeError, IndexError):
                continue
        return target_mask

    def __getitem__(self, index):
        max_retries = 10
        for retry in range(max_retries):
            actual_index = (index + retry) % self.length
            img_path, mask_path = self.samples[actual_index]
            file_name = os.path.basename(img_path)

            try:
                img      = Image.open(img_path).convert("L")
                mask_pil = Image.open(mask_path).convert("L")
            except Exception as e:
                print(f"WARNING: Failed to load {img_path}: {e}")
                continue

            # 从 mask 提取 bbox（连通域分析）
            boxes = self._boxes_from_mask(np.array(mask_pil))

            # mask 转为 PIL '1' 模式（二值）
            mask = mask_pil.point(lambda p: 255 if p > 128 else 0).convert("1")

            # ---------- 数据增强 / padding ----------
            if self.pad_only:
                if self.type == "train" and self.augment:
                    if self.flip_augmentation and random.random() < 0.5:
                        img  = img.transpose(Image.FLIP_LEFT_RIGHT)
                        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                        if len(boxes) > 0:
                            boxes = boxes.copy()
                            boxes[:, 0] = img.width - boxes[:, 0] - boxes[:, 2]
                    if self.vflip_augmentation and random.random() < 0.5:
                        img  = img.transpose(Image.FLIP_TOP_BOTTOM)
                        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                        if len(boxes) > 0:
                            boxes = boxes.copy()
                            boxes[:, 1] = img.height - boxes[:, 1] - boxes[:, 3]
                img, mask, boxes = self._pad_to_target(img, mask, boxes)

            # ---------- 颜色增强（仅训练）----------
            if self.type == "train" and self.augment:
                img_arr = np.array(img, dtype=np.float32) / 255.0
                if self.brightness_jitter > 0:
                    img_arr = random_brightness(img_arr, self.brightness_jitter)
                if self.contrast_jitter > 0:
                    img_arr = random_contrast(img_arr, self.contrast_jitter)
                if self.gaussian_noise > 0:
                    img_arr = random_gaussian_noise(img_arr, self.gaussian_noise)
                img = Image.fromarray(np.clip(img_arr * 255, 0, 255).astype(np.uint8))

            # ---------- 归一化（ImageNet 灰度通道参数）----------
            img_arr    = np.array(img, dtype=np.float32) / 255.0
            img_arr    = (img_arr - 0.485) / 0.229
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float()

            mask_tensor = torch.from_numpy(
                np.array(mask, dtype=np.float32)
            ).unsqueeze(0).float()

            target_mask  = self._create_target_mask(boxes, img_tensor.shape[2], img_tensor.shape[1])
            boxes_tensor = (
                torch.from_numpy(boxes).float()
                if len(boxes) > 0
                else torch.zeros(0, 4, dtype=torch.float32)
            )
            target = {
                "boxes":       boxes_tensor,
                "labels":      torch.zeros(boxes_tensor.size(0), dtype=torch.long),
                "image_id":    torch.tensor(actual_index),
                "target_mask": target_mask,
                "mask":        mask_tensor,
                "file_name":   file_name,
            }
            return img_tensor, target

        # 所有重试均失败，返回 dummy
        print(f"ERROR: NUDTMIRSDTDataset failed to load sample {index}, returning dummy")
        dummy_h = self.pad_to_size[0] if self.pad_to_size else self.image_size
        dummy_w = self.pad_to_size[1] if self.pad_to_size else self.image_size
        img_tensor = torch.zeros(1, dummy_h, dummy_w, dtype=torch.float32)
        target = {
            "boxes":       torch.zeros(0, 4, dtype=torch.float32),
            "labels":      torch.zeros(0, dtype=torch.long),
            "image_id":    torch.tensor(index),
            "target_mask": torch.zeros(1, dummy_h, dummy_w, dtype=torch.float32),
            "mask":        torch.zeros(1, dummy_h, dummy_w, dtype=torch.float32),
            "file_name":   "dummy",
        }
        return img_tensor, target
