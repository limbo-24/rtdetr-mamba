from ultralytics import RTDETR
import torch

try:
    # 1. 加载我们定义好的 YAML
    model = RTDETR("ultralytics/cfg/models/rt-detr/wf-didnet.yaml")
    print("✅ 模型结构解析成功！")
    
    # 2. 尝试一次前向传播 (Forward Pass)
    img = torch.randn(1, 3, 640, 640)
    results = model.predict(img)
    print("✅ 前向推理测试成功！骨干网络改造完成。")
    
except Exception as e:
    print(f"❌ 出错了: {e}")
    

