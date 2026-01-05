import os
import json
from tqdm import tqdm
from PIL import Image

# 配置区
IMG_DIR = '/root/autodl-tmp/afo/images/'  # 原图路径
LABEL_DIR = '/root/autodl-tmp/afo/6categories/'  # 改这里！如果是6分类就改成6categories/
SAVE_DIR = '/root/autodl-tmp/afo/'  # 保存json的位置

TRAIN_LIST = '/root/autodl-tmp/afo/train.txt'
VAL_LIST = '/root/autodl-tmp/afo/validation.txt'
TEST_LIST = '/root/autodl-tmp/afo/test.txt'

CATEGORY_NAMES = ['human','wind/sup-board','boat','bouy','sailboat','kayak']  # 这里是6类（6categories）
# 如果你用6分类，这里就改成6类名字，比如 ['human', 'boat', 'buoy', ...]

def read_image_list(list_file):
    with open(list_file, 'r') as f:
        imgs = f.read().strip().split()
    return imgs

def yolo2coco(img_list, json_save_path, img_dir, label_dir, categories):
    images = []
    annotations = []
    categories_coco = []
    ann_id = 1
    img_id = 1

    name2id = {name: idx for idx, name in enumerate(categories)}

    for idx, name in enumerate(categories):
        categories_coco.append({
            'id': idx,
            'name': name,
            'supercategory': 'none'
        })

    for img_name in tqdm(img_list):
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            continue

        img = Image.open(img_path)
        width, height = img.size

        images.append({
            'file_name': img_name,
            'height': height,
            'width': width,
            'id': img_id
        })

        with open(label_path, 'r') as f:
            labels = f.readlines()

        for label in labels:
            parts = label.strip().split()
            category = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            xmin = (xc - w/2) * width
            ymin = (yc - h/2) * height
            box_w = w * width
            box_h = h * height

            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': category,  # 注意！这里直接拿YOLO里面的ID
                'bbox': [xmin, ymin, box_w, box_h],
                'area': box_w * box_h,
                'iscrowd': 0
            })
            ann_id += 1

        img_id += 1

    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': categories_coco
    }

    with open(json_save_path, 'w') as f:
        json.dump(coco_format, f, indent=4)
    print(f"✔️ 保存成功：{json_save_path}")

if __name__ == "__main__":
    # 读列表
    train_imgs = read_image_list(TRAIN_LIST)
    val_imgs = read_image_list(VAL_LIST)
    test_imgs = read_image_list(TEST_LIST)

    # 转换
    yolo2coco(train_imgs, os.path.join(SAVE_DIR, 'train.json'), IMG_DIR, LABEL_DIR, CATEGORY_NAMES)
    yolo2coco(val_imgs, os.path.join(SAVE_DIR, 'val.json'), IMG_DIR, LABEL_DIR, CATEGORY_NAMES)
    yolo2coco(test_imgs, os.path.join(SAVE_DIR, 'test.json'), IMG_DIR, LABEL_DIR, CATEGORY_NAMES)
