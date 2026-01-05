import sys
import argparse
import os
from ultralytics import RTDETR

'''
训练AFO数据集专用
用法示例：
python train_rtdetr_afo.py --cfg ultralytics/cfg/models/rt-detr/rtdetr-l.yaml --data configs/afo.yaml
'''

def main(opt):
    model = RTDETR(opt.cfg)  # 加载模型结构

    model.info()  # 打印模型信息

    results = model.train(
        data=opt.data,         # 使用自定义数据集yaml
        epochs=50,             # 训练轮数
        imgsz=800,             # 输入尺寸（小目标）
        batch=16,              # 批大小
        workers=4,             # dataloader线程
        device=0,              # 用GPU0
        lr0=0.0002,            # 初始学习率
        optimizer='AdamW',     # 优化器
        close_mosaic=True,     # 关闭mosaic增强（航拍小目标推荐关掉）
    )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/rt-detr/rtdetr-l.yaml', help='模型结构yaml')
    parser.add_argument('--data', type=str, default='configs/afo.yaml', help='数据集配置yaml')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
