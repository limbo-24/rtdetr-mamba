import torch
from ultralytics import RTDETR

def train_cascade_pipeline():
    # =====================================================================
    # 1. 基础配置
    # =====================================================================
    # 指向你刚才修改好的 YAML (包含去雾头 + PGM + RT-DETR)
    model_yaml = 'ultralytics/cfg/models/rt-detr/wf-didnet.yaml'
    
    # 指向检测数据集 (RTTS 或 COCO 格式的 yaml)
    # 如果你是跑 RTTS，确保 rtts.yaml 里路径是对的
    data_yaml = 'ultralytics/cfg/datasets/rtts.yaml' 
    
    # 指向你预训练好的去雾权重 (Epoch 26)
    dehaze_checkpoint = 'runs/dehaze_pretrain_epoch_26.pt'
    
    # 项目保存路径
    project_name = 'runs/detect_train'
    exp_name = 'rtdetr_pgm_cascade'

    # =====================================================================
    # 2. 初始化模型
    # =====================================================================
    print(f"🏗️ 正在从 {model_yaml} 构建级联模型...")
    # 这会随机初始化所有层 (包括 Backbone, PGM, Head, DehazeHead)
    model = RTDETR(model_yaml)

    # =====================================================================
    # 3. 注入“去雾分支”的预训练权重
    # =====================================================================
    if dehaze_checkpoint:
        print(f"💉 正在注入去雾权重: {dehaze_checkpoint}")
        try:
            # 1. 加载 checkpoint
            chkpt = torch.load(dehaze_checkpoint, map_location='cpu')
            
            # 2. 智能提取参数字典
            state_dict = None
            if isinstance(chkpt, dict) and 'model' in chkpt:
                # 情况 A: chkpt['model'] 是整个模型对象 (包含 .float() 方法)
                if hasattr(chkpt['model'], 'state_dict'):
                    state_dict = chkpt['model'].float().state_dict()
                # 情况 B: chkpt['model'] 只是参数字典 (OrderedDict)
                else:
                    state_dict = chkpt['model']
            else:
                # 情况 C: chkpt 本身就是参数字典
                state_dict = chkpt
                
            # 3. 🔥 核心过滤逻辑 (只保留 Layer 0) 🔥
            # 只加载 model.0 (去雾头) 的参数，忽略不匹配的 Backbone/Head
            dehaze_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.0.') or k.startswith('0.'):
                    dehaze_dict[k] = v
            
            if len(dehaze_dict) == 0:
                print("⚠️ 警告: 在 Checkpoint 中未找到去雾头(Layer 0)的权重！")
            else:
                # 4. 加载过滤后的权重 (strict=False)
                model.model.load_state_dict(dehaze_dict, strict=False)
                print(f"✅ 成功仅注入 Layer 0 (去雾头) 权重! (已忽略旧 Backbone)")
                print(f"   - 注入参数量: {len(dehaze_dict)} 个 Tensor")
            
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            import traceback
            traceback.print_exc()
            return

    # =====================================================================
    # 4. 冻结去雾分支 (Freeze Dehaze Head)
    # =====================================================================
    print("\n❄️ 正在冻结去雾分支，解锁 PGM 和检测头...")
    
    frozen_layers = []
    trainable_layers = []
    
    for name, param in model.model.named_parameters():
        # 冻结条件：名字里包含 'HighResMambaDehazeHead'
        # 注意：不要冻结 'PhysicalGuidanceModule'，因为它是新加的，需要训练！
        if 'HighResMambaDehazeHead' in name:
            param.requires_grad = False
            frozen_layers.append(name)
        else:
            param.requires_grad = True # PGM, Backbone, Neck, Head 都要练
            trainable_layers.append(name)

    print(f"   🔒 已冻结层数: {len(frozen_layers)} (Mamba 去雾头)")
    print(f"   🔓 训练层数: {len(trainable_layers)} (PGM + RT-DETR)")

    # =====================================================================
    # 5. 开始全量训练
    # =====================================================================
    print(f"\n🚀 开始端到端级联训练 (Cascade Training)...")
    print(f"   - 策略: 去雾特征(冻结) --> PGM清洗 --> 检测(训练)")
    
    # 显存警告：PGM 和 640 分辨率比较吃显存，如果 OOM，请把 batch 改小 (e.g. 4 -> 2)
    # model.train(
    #     data=data_yaml,
    #     epochs=30,          # 建议跑 50-100 轮
    #     imgsz=224,           # 必须用 640，为了配合 Mamba 的最佳感受野
    #     batch=1,             # 根据你的显存调整 (RT-DETR 比较重)
    #     lr0=0.0001,          # 初始学习率
    #     project=project_name,
    #     name=exp_name,
    #     device='0',          # 指定 GPU
    #     amp=True,            # 开启混合精度加速
    #     plots=True           # 画出训练曲线
    # )
    
    model.train(
    data=data_yaml,
    epochs=3,
    imgsz=128,        # 不要 640，Mamba 扛不住
    lr0=1e-4,
    device=0,
    amp=True,
    workers=4,
    cache=False,      # 关键：减少显存
    plots=False       # 关键：少存中间图
    )


if __name__ == "__main__":
    train_cascade_pipeline()
