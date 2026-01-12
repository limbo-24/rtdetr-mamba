import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

# 尝试导入 Mamba 核心算子，如果没有安装会报错提示
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    from mamba_ssm.modules.mamba_simple import Block as MambaBlock
    IS_MAMBA_AVAILABLE = True
except ImportError:
    IS_MAMBA_AVAILABLE = False
    print("⚠️ 警告: 未检测到 mamba_ssm 库。VSSBlock 将无法运行。")
    print("👉 请运行: pip install mamba-ssm causal-conv1d>=1.2.0")

# -------------------------------------------------------------------------
# 1. 基础工具: LayerNorm_NHWC (解决 Ultralytics NCHW 与 Mamba NHWC 的冲突)
# -------------------------------------------------------------------------
class LayerNorm_NHWC(nn.Module):
    """
    专门给 Mamba 用的 LayerNorm。
    Ultralytics 数据流是 [B, C, H, W]，Mamba 需要 [B, H, W, C]。
    这个层负责在进入 Mamba 前转换维度，做完 Norm 后保持 NHWC 给 Mamba 吃。
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 输入 x: [B, C, H, W] -> 转为 [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class Permute(nn.Module):
    """ 辅助层：把 [B, H, W, C] 转回 [B, C, H, W] """
    def __init__(self, *args):
        super().__init__()
        self.dims = args
    def forward(self, x):
        return x.permute(*self.dims)

# -------------------------------------------------------------------------
# 2. 频域模块: DWT (已验证)
# -------------------------------------------------------------------------
class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False 

    def forward(self, x):
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

# -------------------------------------------------------------------------
# 3. 核心组件: SS2D & VSSBlock (Mamba 核心)
# -------------------------------------------------------------------------
class SS2D(nn.Module):
    """
    2D Selective Scan (VMamba 核心算子)
    将 2D 图像展开为 4 个方向的序列，送入 Mamba SSM 进行扫描。
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # 1. 输入投影 (In_Proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 2. 2D 卷积 (Conv2d) - 用于处理局部关系
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # 3. SSM 核心参数 (A, D, dt, B, C)
        # 这里简化处理，直接定义投影层，利用 mamba_ssm 的底层优化
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(4) # 4个方向
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj # 删除 list，只保留 parameter 以便 saving

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4)
            for _ in range(4)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # A 和 D 参数
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        # 4. 输出投影
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @property
    def dt_rank(self):
        return math.ceil(self.d_model / 16)

    def dt_init(self, dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        m = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(m.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(m.weight, -dt_init_std, dt_init_std)
        
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            m.bias.copy_(inv_dt)
        return m

    def A_log_init(self, d_state, d_inner, copies=1, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    def D_init(self, d_inner, copies=1, merge=True):
        D = torch.ones(d_inner)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (z: gate, x: info)

        # Permute for Conv2d: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) 
        
        # Cross Scan (展开成 4 个方向的序列)
        # 这里简化实现，实际应用中建议使用 mamba_ssm 的 selective_scan_fn 配合自定义的 scan
        # 为了演示和确保运行，我们只使用标准的 Mamba 扫描逻辑，或者如果安装了 cuda 算子则调用
        
#         if IS_MAMBA_AVAILABLE:
#             # 这里是简化的调用逻辑，真实的 SS2D 涉及复杂的 cross_scan/merge
#             # 为保证在 Ultralytics 里能跑，我们将 Feature Map 展平为 Sequence
#             x_flat = x.flatten(2).transpose(1, 2) # [B, L, C]
            
#             # TODO: 真正完全的 4-direction scan 需要较多代码，
#             # 暂时用 1-direction 标准 mamba 替代以跑通流程，性能影响有限
#             # 后续可参考 VMamba 源码补全 cross_scan
#             x_mamba = mamba_inner_fn(
#                 x_flat, 
#                 self.x_proj_weight[0], self.dt_projs_weight[0], 
#                 self.A_logs[0:self.d_inner], self.Ds[0:self.d_inner],
#                 delta_bias=self.dt_projs_bias[0],
#                 delta_softplus=True
#             )
#             y = x_mamba
            
#             # Reshape back: [B, L, C] -> [B, H, W, C]
#             y = y.view(B, H, W, -1)
#         else:
#             # Fallback: 如果没装 Mamba，用 Identity 避免报错，仅做 Conv
#             y = x.permute(0, 2, 3, 1)

#         y = self.out_norm(y)
#         y = y * F.silu(z)
#         out = self.out_proj(y)
#         if self.dropout is not None:
#             out = self.dropout(out)
#         return out
    
        # 3. 核心 Mamba 扫描分支
        if IS_MAMBA_AVAILABLE:
            x_flat = x.flatten(2) # [B, C, L]
            
            # 使用更稳健的线性投影方式
            import torch.nn.functional as F
            
            # 这里的索引必须极其小心，确保不丢失维度
            # 假设 x_proj_weight 是 ParameterList 或 3D Tensor
            proj_weight = self.x_proj_weight[0] 
            x_dbl = F.linear(x_flat.transpose(1, 2), proj_weight) 
            
            # 1. 拆分 dt, B, C
            dt, B_vec, C_vec = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            
            # 2. 投影 dt
            dt = F.linear(dt, self.dt_projs_weight[0]) 
            dt = dt.transpose(1, 2).contiguous()
            
            # 3. 准备 A, B, C, D (注意这里恢复切片逻辑 [0:self.d_inner])
            # 这样能保证 A 是 [d_inner, d_state] 的 2D 形状
            A = -torch.exp(self.A_logs[0:self.d_inner].float()) 
            B_vec = B_vec.transpose(1, 2).contiguous()
            C_vec = C_vec.transpose(1, 2).contiguous()
            D = self.Ds[0:self.d_inner].float()
            
            # 4. 调用算子
            y = selective_scan_fn(
                x_flat, 
                dt, 
                A, B_vec, C_vec, D, 
                z=None, 
                delta_bias=self.dt_projs_bias[0].float(),
                delta_softplus=True,
                return_last_state=False
            )
            
            y = y.transpose(1, 2).view(B, H, W, -1)
        else:
            y = x.permute(0, 2, 3, 1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    """
    Ultralytics 适配版 VSS Block
    """
    def __init__(self, hidden_dim, d_state=16):
        super().__init__()
        self.ln_1 = LayerNorm_NHWC(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state)
        self.ln_2 = nn.LayerNorm(hidden_dim) # FFN 前的 Norm
        
        # FFN (Feed Forward Network)
        self.ffn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, 1), # Point-wise
            nn.GELU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        input_x = x
        
        # 1. VSSM 部分 (需处理 NHWC)
        x_norm = self.ln_1(x) # 变成 [B, H, W, C]
        x_vss = self.self_attention(x_norm) # 输出 [B, H, W, C]
        x_vss = x_vss.permute(0, 3, 1, 2) # 转回 [B, C, H, W]
        x = input_x + x_vss 

        # 2. FFN 部分 (Conv 实现，直接吃 NCHW)
        # 注意: 如果用 nn.LayerNorm，需要 permute 维度。这里 ln_2 是标准 LayerNorm
        x_norm2 = x.permute(0, 2, 3, 1) # [B, H, W, C]
        x_norm2 = self.ln_2(x_norm2)
        x_norm2 = x_norm2.permute(0, 3, 1, 2) # [B, C, H, W]
        
        x_ffn = self.ffn(x_norm2)
        x = x + x_ffn
        
        return x

# -------------------------------------------------------------------------
# 4. 新增: MCAM (YOLO-Extreme) - 替换 SimAM
# -------------------------------------------------------------------------
class MCAM(nn.Module):
    """
    Multi-Dimensional Collaborative Attention Module
    Ref: YOLO-Extreme
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel Branch
        self.fc_channel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Height Branch (H-Attention)
        self.conv_h = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid_h = nn.Sigmoid()
        
        # Width Branch (W-Attention)
        self.conv_w = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        b, c, h, w = x.size()
        y_c = self.fc_channel(self.pool(x)) # [B, C, 1, 1]
        
        # Spatial Attention (Simplified MCAM for efficiency)
        # Height map
        x_h = x.mean(dim=3, keepdim=True) # [B, C, H, 1]
        a_h = self.sigmoid_h(self.conv_h(x_h)) # [B, 1, H, 1]
        
        # Width map
        x_w = x.mean(dim=2, keepdim=True) # [B, C, 1, W]
        a_w = self.sigmoid_w(self.conv_w(x_w)) # [B, 1, 1, W]
        
        return x * y_c * a_h * a_w

# # -------------------------------------------------------------------------
# # 5. 去雾头: HighResMambaDehazeHead
# # -------------------------------------------------------------------------
# class HighResMambaDehazeHead(nn.Module):
#     def __init__(self, in_ch=128, d_state=16, depth=2):
#         super().__init__()
#         self.proj = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1), nn.BatchNorm2d(in_ch), nn.SiLU())
        
#         # 使用真实的 VSSBlock
#         self.mamba_layers = nn.Sequential(*[
#             VSSBlock(hidden_dim=in_ch, d_state=d_state) for _ in range(depth)
#         ])
        
#         self.mid_conv = nn.Sequential(nn.Conv2d(in_ch, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
#         self.t_head = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        
#         # 重构分支 (训练时用)
#         self.recon = nn.Sequential(
#             self._pixel_shuffle_block(64, 32), # H/8
#             self._pixel_shuffle_block(32, 16), # H/4
#             self._pixel_shuffle_block(16, 16), # H/2
#             self._pixel_shuffle_block(16, 8),  # H
#             nn.Conv2d(8, 3, 3, 1, 1),
#             nn.Sigmoid()
#         )

#     def _pixel_shuffle_block(self, in_c, out_c):
#         return nn.Sequential(
#             nn.Conv2d(in_c, out_c * 4, 3, 1, 1),
#             nn.PixelShuffle(2),
#             nn.ReLU(inplace=True)
#         )

# #     def forward(self, x_ll):
# #         x = self.proj(x_ll)
# #         x = self.mamba_layers(x)
# #         feat = self.mid_conv(x)
# #         t_map = self.t_head(feat)
        
# #         recon_img = None
# #         if self.training:
# #             recon_img = self.recon(feat)
            
# #         return t_map, recon_img, feat
    
#     def forward(self, x_ll, batch=None):
#         # 1. 兼容性处理：如果输入是列表（来自 tasks.py 的修改），解包取出第一个元素
#         if isinstance(x_ll, list):
#             x_ll = x_ll[0]
            
#         # 2. 正常的去雾逻辑
#         x = self.proj(x_ll)
#         x = self.mamba_layers(x)
#         feat = self.mid_conv(x)
#         t_map = self.t_head(feat)
        
#         # 3. 重构分支
#         recon_img = None
#         if self.training:
#             recon_img = self.recon(feat)
            
#         # 返回结果
#         return t_map, recon_img, feat

# 5. 去雾头: HighResMambaDehazeHead
# -------------------------------------------------------------------------
class HighResMambaDehazeHead(nn.Module):
    def __init__(self, in_ch=128, d_state=16, depth=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1), nn.BatchNorm2d(in_ch), nn.SiLU())
        
        # 使用真实的 VSSBlock
        self.mamba_layers = nn.Sequential(*[
            VSSBlock(hidden_dim=in_ch, d_state=d_state) for _ in range(depth)
        ])
        
        self.mid_conv = nn.Sequential(nn.Conv2d(in_ch, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.t_head = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        
        # 重构分支
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

    def forward(self, x_ll, batch=None):
        # 1. 兼容性处理：如果输入是列表，解包取出第一个元素
        if isinstance(x_ll, list):
            x_ll = x_ll[0]
            
        # 2. 正常的去雾逻辑
        x = self.proj(x_ll)
        x = self.mamba_layers(x)
        feat = self.mid_conv(x)
        t_map = self.t_head(feat)
        
        # 3. 重构分支 (修复：去掉 self.training 判断，确保验证时也能输出图像)
        recon_img = self.recon(feat)  # <--- 这里改了
            
        # 返回结果 (Tuple: 透射率, 恢复图, 特征)
        return t_map, recon_img, feat

# -------------------------------------------------------------------------
# 6. 交互模块
# -------------------------------------------------------------------------
class PhysicalGuidanceModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fusion = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU())

    def forward(self, feat_high, t_map):
        t_map_up = F.interpolate(t_map, size=feat_high.shape[2:], mode='bilinear', align_corners=False)
        feat_weighted = feat_high * t_map_up 
        return self.fusion(feat_high + feat_weighted)

class SemanticFeedbackModule(nn.Module):
    def __init__(self, dehaze_dim=64, det_dim=256):
        super().__init__()
        self.adapter = nn.Conv2d(det_dim, dehaze_dim, 1)
        self.fusion = nn.Sequential(nn.Conv2d(dehaze_dim*2, dehaze_dim, 3, 1, 1), nn.SiLU())

    def forward(self, dehaze_feat, det_feat):
        det_small = F.interpolate(self.adapter(det_feat), size=dehaze_feat.shape[2:], mode='bilinear', align_corners=False)
        return self.fusion(torch.cat([dehaze_feat, det_small], dim=1))

# -------------------------------------------------------------------------
# 7. 无参注意力: SimAM (补全缺失的模块)
# -------------------------------------------------------------------------
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

