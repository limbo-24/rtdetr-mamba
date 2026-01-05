import argparse
import os
from ultralytics import RTDETR

"""
测试AFO数据集训练好的RT-DETR模型
用法示例：
python predict_rtdetr.py --weights runs/detect/train14/weights/best.pt --source datasets/fog
"""

def main(opt):
    # 1. 加载训练好的模型权重
    model = RTDETR(opt.weights)

    # 2. 推理（source 可以是图片/视频/文件夹）
    results = model.predict(
        source=opt.source,   # 输入图片/文件夹/视频
        imgsz=800,           # 推理输入尺寸（要和训练时一致）
        conf=0.5,           # 置信度阈值
        save=True,           # 保存结果
        save_txt=True,       # 保存 txt 格式结果 (YOLO格式：class x y w h)
        save_conf=True       # 保存置信度到 txt
    )

    # 3. 打印检测结果
    for r in results:
        print(r)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--source', type=str, required=True, help='测试数据路径，可以是图片/文件夹/视频')

    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
