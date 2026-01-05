import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# ================= 配置区域 =================
# 1. 这里引入你定义模型的类
# 假设你的代码结构是 rtdetr_pytorch/models/...
# 你需要确保能 import 到 RTDETR 类
import sys
sys.path.append('./') # 将当前目录加入路径，防止找不到模块
# from models.rtdetr import build_rtdetr # <--- 请根据你的实际代码路径修改这里

# 2. 模型权重路径
MODEL_CKPT = "path/to/your/trained_rtdetr.pth" 

# 3. 测试图片路径 (建议使用经过 WDMamba 去雾后的图片)
IMG_PATH = "download (1).jpg"

# ===========================================

def get_rtdetr_model(ckpt_path):
    """
    加载模型并载入权重
    """
    # 1. 初始化模型结构 (这里需要根据你的 config 参数来实例化)
    # 如果你有 config 文件，最好用你的 build_function
    # model = build_rtdetr(args) 
    
    # 临时模拟：这里假设你已经实例化好了 model 对象
    # 你可以在这里粘贴你 main.py 里构建模型的代码
    # model = ... 
    
    # 2. 加载权重
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # RT-DETR 的权重通常保存在 'model' 或 'ema' 键里
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # 处理一下可能的 key 匹配问题 (去掉 module. 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # model.load_state_dict(new_state_dict) # 加载权重
    model.eval()
    return model

def run_cam():
    # 1. 准备模型
    # 注意：为了运行演示，你需要确保 model 变量已经正确加载
    # model = get_rtdetr_model(MODEL_CKPT)
    
    # --- 临时：如果你还没调通模型加载，可以用 torchvision 的 resnet 做测试 ---
    from torchvision.models import resnet50
    model = resnet50(pretrained=True)
    # -------------------------------------------------------------

    model.eval()

    # 2. 选择目标层 (Target Layer)
    # 对于 RT-DETR (使用 ResNet Backbone)，通常看 Layer4 的最后一层
    # 如果你加了 SimAM，可能叫 model.backbone.layer4[-1] 或者 model.backbone.simam
    # 请打印 print(model) 确认层级名称
    
    # 针对标准 ResNet Backbone:
    target_layers = [model.layer4[-1]] 
    # 针对 RT-DETR 真实结构可能是: [model.backbone.body.layer4[-1]]
    
    # 3. 准备图像
    rgb_img = cv2.imread(IMG_PATH)[:, :, ::-1] # BGR -> RGB
    rgb_img = np.float32(rgb_img) / 255
    # 标准 ImageNet 归一化
    input_tensor = preprocess_image(rgb_img, 
                                   mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])

    # 4. 初始化 CAM 工具
    # 使用 EigenCAM！
    # 为什么？因为 GradCAM 需要 loss.backward()，而 RT-DETR 输出是字典，
    # 且目标检测很难定义单一的 class target。
    # EigenCAM 直接计算特征图的主成分，能完美展示"网络觉得哪里重要(物体)"，无需反向传播。
    cam = EigenCAM(model=model, target_layers=target_layers)

    # 5. 生成热力图
    # targets=None 表示不需要特定的分类目标，EigenCAM 会自动寻找显著区域
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    # 取第一张图 (Batch size = 1)
    grayscale_cam = grayscale_cam[0, :]

    # 6. 叠加显示
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 7. 保存结果
    save_name = "attention_map_baseline.jpg"
    cv2.imwrite(save_name, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Attention map saved to {save_name}")

if __name__ == '__main__':
    run_cam()