# codes to make visualization of CNN weights.
import mynn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 加载训练好的CNN模型
model = nn.models.Model_CNN(input_shape=(1, 28, 28), num_classes=10)
model.load_model(r'.\cnn_models\CNN_2.pickle')  # 修改为你的CNN模型路径

# 提取第一个卷积层的权重
conv1_weights = model.layers[0].params['W']  # 形状为(out_channels, in_channels, kH, kW)
print("Conv1 weights shape:", conv1_weights.shape)


# 可视化第一个卷积层的滤波器
def plot_conv_weights(weights, ncols=8):
    # 参数处理
    out_channels, in_channels, kH, kW = weights.shape
    nrows = int(np.ceil(out_channels / ncols))

    # 创建画布
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = axes.ravel()

    # 绘制每个滤波器
    for i in range(out_channels):
        # 取第i个输出通道的权重（对于MNIST输入通道为1）
        kernel = weights[i, 0]  # 形状(kH, kW)
        axes[i].imshow(kernel, cmap='viridis', interpolation='none')
        axes[i].set_title(f'Kernel {i + 1}')
        axes[i].axis('off')

    # 隐藏多余的子图
    for j in range(out_channels, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# 可视化第一个卷积层的32个滤波器
plot_conv_weights(conv1_weights, ncols=8)

# 可视化第二个卷积层的滤波器（可选）
if len(model.layers) >= 4 and isinstance(model.layers[3], nn.op.conv2D):
    conv2_weights = model.layers[3].params['W']
    print("Conv2 weights shape:", conv2_weights.shape)
    plot_conv_weights(conv2_weights, ncols=8)