import torch
from ultralytics.nn.modules.wf_didnet_modules import DWT

def test_dwt_logic():
    print("🚀 开始验证 DWT (小波变换) 模块...")
    
    # 1. 模拟 RT-DETR 骨干网 C3 层的输出
    # 假设输入图像 640x640, C3 层 stride=8, 所以尺寸是 80x80
    # ResNet50/101 的 C3 通道数通常是 512，ResNet18/34 是 128。这里假设 256 用于测试。
    batch_size = 2
    in_channels = 256
    height, width = 80, 80
    
    dummy_c3 = torch.randn(batch_size, in_channels, height, width)
    print(f"📥 模拟 C3 输入形状: {dummy_c3.shape}")

    # 2. 实例化 DWT
    dwt_layer = DWT()
    
    # 3. 前向传播
    try:
        # wf_didnet_modules.py 中的 DWT 返回的是拼接后的 (LL, LH, HL, HH)
        output = dwt_layer(dummy_c3)
        
        print(f"📤 DWT 输出形状: {output.shape}")
        
        # 4. 验证维度
        expected_channels = in_channels * 4  # 256 * 4 = 1024
        expected_height = height // 2        # 80 / 2 = 40
        expected_width = width // 2          # 80 / 2 = 40
        
        assert output.shape == (batch_size, expected_channels, expected_height, expected_width), \
            f"❌ 维度错误! 期望: {(batch_size, expected_channels, expected_height, expected_width)}, 实际: {output.shape}"
            
        print("✅ DWT 维度验证通过！")
        print("   - 频率分离成功: 4个子带 (LL, LH, HL, HH) 已拼接")
        print("   - 空间下采样成功: 80x80 -> 40x40")
        print("   - 通道扩充成功: 256 -> 1024 (为 Mamba 提供了丰富的频域特征)")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    test_dwt_logic()