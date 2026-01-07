import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from ultralytics import RTDETR # 确保路径在 PythonPath 中

# ---------------------------------------------------------
# 1. 结构相似性损失 (SSIM) - 保护物体轮廓
# ---------------------------------------------------------
def ssim(img1, img2, window_size=11, size_average=True):
    # 这是一个简化版的 SSIM 实现，确保不依赖额外库
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# ---------------------------------------------------------
# 2. Haze4K 数据集类
# ---------------------------------------------------------
class Haze4KDataset(Dataset):
    def __init__(self, root_dir, split='train', size=640):
        self.hazy_path = os.path.join(root_dir, split, 'hazy')
        self.clear_path = os.path.join(root_dir, split, 'clear')
        self.filenames = [f for f in os.listdir(self.hazy_path) if f.endswith(('.png', '.jpg'))]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        fname = self.filenames[index]
        hazy_img = self.transform(Image.open(os.path.join(self.hazy_path, fname)).convert('RGB'))
        clear_img = self.transform(Image.open(os.path.join(self.clear_path, fname)).convert('RGB'))
        return hazy_img, clear_img

    def __len__(self):
        return len(self.filenames)

# ---------------------------------------------------------
# 3. 预训练主循环
# ---------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    yaml_path = 'ultralytics/cfg/models/rt-detr/wf-didnet.yaml'
    # 使用 RTDETR 包装器加载，它会自动触发 tasks.py 里的 parse_model
    model_wrapper = RTDETR(yaml_path)
    model = model_wrapper.model.to(device)

    # 数据加载
    train_ds = Haze4KDataset('/root/autodl-tmp/Haze4K', split='train')
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion_l1 = nn.L1Loss()
    
    print("🚀 开始单独预训练去雾分支...")
    model.train()

    for epoch in range(30): # 预训练 30 轮通常足够
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for hazy, clear in pbar:
            hazy, clear = hazy.to(device), clear.to(device)
            
            optimizer.zero_grad()
            
            # 核心：传入 mode='train_dehaze' 触发 tasks.py 里的拦截逻辑
            recon_img = model(hazy, mode='train_dehaze')
            
            # 混合 Loss
            loss_l1 = criterion_l1(recon_img, clear)
            loss_ssim = 1 - ssim(recon_img, clear)
            total_loss = loss_l1 + 0.2 * loss_ssim # SSIM 权重设为 0.2
            
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix(Loss=f"{total_loss.item():.4f}", L1=f"{loss_l1.item():.4f}")

        # 保存预训练权重
        save_path = f"runs/dehaze_pretrain_epoch_{epoch}.pt"
        os.makedirs("runs", exist_ok=True)
        torch.save({'model': model.state_dict()}, save_path)

if __name__ == "__main__":
    train()