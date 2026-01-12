import torch
from ultralytics import RTDETR
import os

def main():
    # ---------------------------------------------------------
    # 1. 路径配置
    # ---------------------------------------------------------
    # 你的模型配置文件路径
    model_yaml = 'ultralytics/cfg/models/rt-detr/wf-didnet.yaml'
    # 你的目标检测数据集配置文件 (包含 train/val 路径和类别信息)
    # 例如：'ultralytics/cfg/datasets/coco8.yaml' 或你自己的 afo.yaml
    data_yaml = 'ultralytics/cfg/datasets/coco8.yaml' 
    
    # 你刚刚预训练好的去雾权重路径
    pretrained_dehaze_weights = 'runs/dehaze_pretrain_epoch_29.pt'
    
    device = '0' if torch.cuda.is_available() else 'cpu'

    # ---------------------------------------------------------
    # 2. 初始化模型并加载去雾预训练权重
    # ---------------------------------------------------------
    print(f"🏗️ 正在初始化模型架构: {model_yaml}")
    model = RTDETR(model_yaml)
    
    if os.path.exists(pretrained_dehaze_weights):
        print(f"💉 正在注入去雾预训练权重: {pretrained_dehaze_weights}")
        checkpoint = torch.load(pretrained_dehaze_weights, map_location='cpu')
        
        # 注意：这里我们只加载 model 的 state_dict
        # strict=False 非常关键，因为它会忽略预训练权重中没有的 '检测头' 参数
        # 从而只同步 Backbone, Neck, DWT 和 Mamba 分支的权重
        model.model.load_state_dict(checkpoint['model'], strict=False)
        print("✅ 权重注入成功！")
    else:
        print("⚠️ 未找到预训练权重，将从零开始训练（不建议）。")

    # ---------------------------------------------------------
    # 3. (可选) 冻结权重策略
    # ---------------------------------------------------------
    # 如果你想让去雾层保持不变，只训练检测头以节省显存：
    # for i, (name, param) in enumerate(model.model.named_parameters()):
    #     if i <= 31: # 第 31 层之前是去雾分支
    #         param.requires_grad = False
    # print("❄️ 已冻结去雾分支，仅训练检测头。")

    # ---------------------------------------------------------
    # 4. 开始目标检测训练
    # ---------------------------------------------------------
    print("🚀 开始目标检测全量训练...")
    model.train(
        data=data_yaml,
        epochs=100,         # 检测训练通常需要更多轮次
        imgsz=640,          # 恢复到 640 分辨率进行检测
        batch=4,            # 根据你的显存调整
        device=device,
        project='runs/detect',
        name='rtdetr_mamba_dehaze',
        optimizer='AdamW',
        lr0=1e-4,           # 初始学习率
        warmup_epochs=3,    # 热身轮次
        # mode='detect'     # 默认就是 detect，会触发 tasks.py 里的检测逻辑
    )

if __name__ == "__main__":
    main()