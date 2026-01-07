import torch
from ultralytics import RTDETR
import gc

def debug_forward_pass():
    print("🛠️ 启动深度调试：验证双分支前向传播 (GPU模式)...")
    
    # 1. 强制清理内存
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 运行设备: {device}")
    
    try:
        # 3. 加载模型
        print("⏳ 正在加载模型配置...")
        # 注意：Ultralytics 默认加载到 CPU，我们需要手动搬运或让它自动处理
        model = RTDETR("ultralytics/cfg/models/rt-detr/wf-didnet.yaml")
        
        # 4. 将底层模型移动到 GPU
        # model.model 是底层的 nn.Module
        nn_model = model.model.to(device)
        nn_model.eval() # 切换到评估模式 (关闭 Dropout/BatchNorm 更新)
        print("✅ 模型已加载并移动到 GPU！")
        
        # 5. 构造虚拟输入 (Batch=1, RGB, 640x640)
        # 注意：如果显存还是不够，可以把 640 改成 320 试试
        img_size = 640
        img = torch.randn(1, 3, img_size, img_size).to(device)
        print(f"📥 输入张量形状: {img.shape}")
        
        # 6. 执行前向传播 (使用 no_grad 节省大量内存)
        print("🔄 执行前向传播 (Forward Pass)...")
        with torch.no_grad(): # <--- 关键！不计算梯度能省下一半显存
            output = nn_model(img)
        
        # 7. 分析输出
        print("\n🔍 输出结果分析:")
        if isinstance(output, tuple):
            print(f"   输出类型: Tuple (长度={len(output)})")
            
            # 这里对应您 yaml 最后的两个 Head：
            # output[-1] 是去雾头 HighResMambaDehazeHead 的输出 (T_map, Recon, Feat)
            # output[-2] 是检测头 RTDETRDecoder 的输出
            
            # 由于 Ultralytics 的 Head 包装机制，输出结构可能会嵌套
            # 我们直接打印每一项的形状来看看
            for i, item in enumerate(output):
                if isinstance(item, torch.Tensor):
                    print(f"   [项 {i}] Tensor: {item.shape}")
                elif isinstance(item, tuple):
                    print(f"   [项 {i}] Tuple 长度: {len(item)}")
                    # 尝试拆解去雾头的 Tuple
                    if len(item) == 3: 
                        t_map, recon, feat = item
                        print(f"      -> 疑似去雾头输出:")
                        print(f"         T_map: {t_map.shape}")
                        if recon is not None: print(f"         Recon: {recon.shape}")
                        print(f"         Feat:  {feat.shape}")
            
            print("\n✅ 双分支数据流打通成功！没有报错即是胜利！")
            
        else:
            # 如果只返回了一个 Tensor，说明结构可能还有问题
            print(f"   输出类型: {type(output)}")
            
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_forward_pass()