# ultralytics/nn/modules/wf_didnet_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -------------------------------------------------------------------------
# 1. 基础组件: DWT (离散小波变换) & IWT
# -------------------------------------------------------------------------
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_LH = -x1 - x2 + x3 + x4
    x_HL = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_LH, x_HL, x_HH), 1)

class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False
    def forward(self, x):
        return dwt_init(x)

# -------------------------------------------------------------------------
# 2. 核心组件: VSSBlock (从你的 WDMamba 代码中精简移植)
#    注意: 这需要你的环境安装了 mamba_ssm 或 selective_scan_cuda
# -------------------------------------------------------------------------
class VSSBlock(nn.Module):
    def __init__(self, hidden_dim, d_state=16):
        super().__init__()
        # 这里为了简化，我写了一个占位符。
        # 请务必从 WDMamba/basicsr/archs/wavemamba_arch.py 中
        # 复制真实的 VSSBlock 实现粘贴到这里！
        # 确保包含 SS2D, VSSM 等依赖类。
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1) # 临时替代
        
    def forward(self, x):
        # 真实实现应该是 Mamba 的操作
        return self.conv(x) + x

# -------------------------------------------------------------------------
# 3. 去雾头: HighResMambaDehazeHead (频域去雾分支)
# -------------------------------------------------------------------------
class HighResMambaDehazeHead(nn.Module):
    def __init__(self, in_ch=128, d_state=16, depth=2):
        super().__init__()
        # 1. 投影层
        self.proj = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1), nn.BatchNorm2d(in_ch), nn.SiLU())
        
        # 2. Mamba 堆叠 (处理低频全局信息)
        self.mamba_layers = nn.Sequential(*[
            VSSBlock(hidden_dim=in_ch, d_state=d_state) for _ in range(depth)
        ])
        
        # 3. 中间层
        self.mid_conv = nn.Sequential(nn.Conv2d(in_ch, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        
        # 4. 分支A: 透射率图 (用于交互)
        self.t_head = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        
        # 5. 分支B: 图像重构 (用于 Loss) - PixelShuffle 上采样
        self.recon = nn.Sequential(
            self._pixel_shuffle_block(64, 32), # H/8
            self._pixel_shuffle_block(32, 16), # H/4
            self._pixel_shuffle_block(16, 16), # H/2
            self._pixel_shuffle_block(16, 8),  # H
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def _pixel_shuffle_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_ll):
        # x_ll: [B, 128, H/16, W/16]
        x = self.proj(x_ll)
        x = self.mamba_layers(x)
        feat = self.mid_conv(x)
        
        t_map = self.t_head(feat)
        
        recon_img = None
        if self.training: # 只有训练时才重构图像，节省显存
            recon_img = self.recon(feat)
            
        return t_map, recon_img, feat # 返回 feat 供后续反馈使用

# -------------------------------------------------------------------------
# 4. 交互模块: PGI & SFM
# -------------------------------------------------------------------------
class PhysicalGuidanceModule(nn.Module):
    """ 去雾 -> 检测 (乘法交互) """
    def __init__(self, in_channels):
        super().__init__()
        self.fusion = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU())

    def forward(self, feat_high, t_map):
        # feat_high: [B, 384, H/8, W/8] (C3的高频部分)
        # t_map: [B, 1, H/16, W/16]
        t_map_up = F.interpolate(t_map, size=feat_high.shape[2:], mode='bilinear', align_corners=False)
        feat_weighted = feat_high * t_map_up 
        return self.fusion(feat_high + feat_weighted)

class SemanticFeedbackModule(nn.Module):
    """ 检测 -> 去雾 (拼接交互) """
    def __init__(self, dehaze_dim=64, det_dim=256):
        super().__init__()
        self.adapter = nn.Conv2d(det_dim, dehaze_dim, 1)
        self.fusion = nn.Sequential(nn.Conv2d(dehaze_dim*2, dehaze_dim, 3, 1, 1), nn.SiLU())

    def forward(self, dehaze_feat, det_feat):
        # det_feat: [B, 256, H/8, W/8] (P3) -> downsample to dehaze size
        det_small = F.interpolate(self.adapter(det_feat), size=dehaze_feat.shape[2:], mode='bilinear', align_corners=False)
        return self.fusion(torch.cat([dehaze_feat, det_small], dim=1))