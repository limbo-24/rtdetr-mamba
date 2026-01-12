import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from ultralytics import RTDETR

def validate_dehaze():
    # --- 1. 配置参数 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_path = 'ultralytics/cfg/models/rt-detr/wf-didnet.yaml'
    # 路径根据你的实际情况修改，优先使用第 26 轮
    weight_path = 'runs/dehaze_pretrain_epoch_26.pt' 
    test_haze_dir = '/root/autodl-tmp/Haze4K/test/haze'
    test_gt_dir = '/root/autodl-tmp/Haze4K/test/gt'
    save_dir = 'runs/val_results'
    os.makedirs(save_dir, exist_ok=True)

    img_size = 640  # 保持和预训练时一致

    # --- 2. 加载模型与权重 ---
    print(f"正在加载模型并注入权重: {weight_path}")
    model_wrapper = RTDETR(yaml_path)
    ckpt = torch.load(weight_path, map_location='cpu')
    model_wrapper.model.load_state_dict(ckpt['model'])
    model = model_wrapper.model.to(device)
    model.eval()

    # --- 3. 准备数据预处理 ---
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    haze_files = sorted([f for f in os.listdir(test_haze_dir) if f.endswith('.png')])
    
    psnr_list = []
    ssim_list = []

    print(f"开始测试 {len(haze_files)} 张图片...")

    # --- 4. 推理循环 ---
    with torch.no_grad():
        for i, h_name in enumerate(tqdm(haze_files)):
            # 匹配文件名 (1000_0.73_1.8.png -> 1000.png)
            img_id = h_name.split('_')[0]
            gt_name = f"{img_id}.png"
            gt_path = os.path.join(test_gt_dir, gt_name)

            if not os.path.exists(gt_path):
                continue

            # 读取并处理图片
            hazy_img_pil = Image.open(os.path.join(test_haze_dir, h_name)).convert('RGB')
            gt_img_pil = Image.open(gt_path).convert('RGB')

            hazy_tensor = transform(hazy_img_pil).unsqueeze(0).to(device)
            # GT 也需要 Resize 到相同尺寸进行计算
            gt_img_resized = gt_img_pil.resize((img_size, img_size), Image.LANCZOS)
            gt_np = np.array(gt_img_resized)

            # 模型前向传播 (触发 tasks.py 里的拦截逻辑)
            # 根据你之前的 tasks.py，mode='train_dehaze' 会返回 recon_img
            recon_tensor = model(hazy_tensor, mode='train_dehaze')
            
            # --- 防御性检查 ---
            if recon_tensor is None:
                print(f"❌ 警告：模型返回了 None！请检查 tasks.py 是否正确拦截了 mode='train_dehaze'")
                continue

           
            # 后处理：Tensor -> Numpy [0, 255]
            recon_img = recon_tensor.squeeze().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
            recon_img = (recon_img * 255).astype(np.uint8)

            # 计算指标 (使用 skimage 标准库)
            cur_psnr = psnr(gt_np, recon_img, data_range=255)
            cur_ssim = ssim(gt_np, recon_img, channel_axis=2, data_range=255)
            
            psnr_list.append(cur_psnr)
            ssim_list.append(cur_ssim)

            # 每隔 50 张保存一张对比图
            if i % 50 == 0:
                # 拼接：左输入 | 中恢复 | 右真值
                comparison = torch.cat([
                    hazy_tensor.cpu().squeeze(), 
                    recon_tensor.cpu().squeeze(), 
                    transform(gt_img_resized)
                ], dim=2)
                save_image(comparison, f"{save_dir}/res_{i}.png")

    # --- 5. 输出最终评分 ---
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    print("\n" + "="*30)
    print(f"📊 Haze4K 测试集最终结果 (Epoch 26):")
    print(f"⭐ Average PSNR: {avg_psnr:.2f} dB")
    print(f"⭐ Average SSIM: {avg_ssim:.4f}")
    print("="*30)
    print(f"对比图已保存至: {save_dir}")

if __name__ == "__main__":
    validate_dehaze()