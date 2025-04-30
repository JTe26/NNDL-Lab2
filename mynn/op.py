from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
        self.params = {}  # 参数初始化（子类可覆盖）
        self.grads = {}
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass
'''

class Layer:
    def __init__(self):
        self.optimizable = True  # 默认所有层可优化（如 Conv2D、Linear）
        self.params = {}  # 参数初始化（子类可覆盖）
        self.grads = {}  # 梯度初始化（子类可覆盖）

    def forward(self, X):
        raise NotImplementedError("子类必须实现 forward 方法")

    def backward(self, grad):
        raise NotImplementedError("子类必须实现 backward 方法")

    def __call__(self, X):
        return self.forward(X)

'''
class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.in_dim = in_dim  # 必须确保有这个属性
        self.out_dim = out_dim
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # 保存输入用于反向传播
        return np.dot(X, self.params['W']) + self.params['b']  # 确保返回非空值

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.dot(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0)

        # 计算传递给前一层的梯度
        dx = np.dot(grad, self.params['W'].T)

        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']

        return dx
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_decay=False,
                weight_decay_lambda=1e-4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding

            # 初始化卷积核和偏置
        kH, kW = self.kernel_size
        self.W = np.random.randn(out_channels, in_channels, kH, kW) * np.sqrt(2. / (in_channels * kH * kW))
        self.b = np.zeros((out_channels, 1))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}

            # L2 正则化配置
        self.weight_decay = weight_decay  # 是否启用 L2 正则化
        self.weight_decay_lambda = weight_decay_lambda  # 正则化强度

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        batch_size, in_channels, H, W = X.shape
        kH, kW = self.kernel_size

        # 计算输出尺寸
        new_H = (H + 2 * self.padding - kH) // self.stride + 1
        new_W = (W + 2 * self.padding - kW) // self.stride + 1

        # 添加 padding
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant')

        # 初始化输出
        output = np.zeros((batch_size, self.out_channels, new_H, new_W))

        # 滑动窗口卷积
        for i in range(new_H):
            for j in range(new_W):
                h_start = i * self.stride
                h_end = h_start + kH
                w_start = j * self.stride
                w_end = w_start + kW

                # 提取输入局部区域 [batch, in_channels, kH, kW]
                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]

                # 计算卷积 [batch, out_channels] += sum(X_slice * W) + b
                output[:, :, i, j] = np.tensordot(X_slice, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b.squeeze()

        self.input = X  # 保存输入用于反向传播
        return output

    def backward(self, grad):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input  # 输入数据 [batch, in_channels, H, W]
        batch_size, in_channels, H, W = X.shape
        out_channels, _, kH, kW = self.W.shape
        stride = self.stride
        padding = self.padding

        # 初始化梯度容器
        dX = np.zeros_like(X)
        dW = np.zeros_like(self.W)
        db = np.sum(grad, axis=(0, 2, 3)).reshape(self.b.shape) / batch_size  # 按批量平均

        # 添加padding以计算输入梯度
        X_padded = np.pad(X,
                          ((0, 0), (0, 0),
                           (padding, padding), (padding, padding)),
                          mode='constant')

        # 计算卷积核梯度 dW ---------------------------------------------------
        for i in range(grad.shape[2]):  # 遍历输出高度
            for j in range(grad.shape[3]):  # 遍历输出宽度
                # 计算当前感受野位置
                h_start = i * stride
                h_end = h_start + kH
                w_start = j * stride
                w_end = w_start + kW

                # 提取输入局部区域 [batch, in_channels, kH, kW]
                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]

                # 梯度累加（注意转置对齐维度）
                dW += np.tensordot(grad[:, :, i, j], X_slice, axes=([0], [0]))

        dW /= batch_size  # 按批量平均梯度

        # 计算输入梯度 dX ----------------------------------------------------
        # 旋转卷积核180度（实现转置卷积）
        W_rot180 = np.rot90(self.W, 2, axes=(2, 3))

        # 添加padding以适应反向卷积操作
        grad_padded = np.pad(grad,
                             ((0, 0), (0, 0),
                              (kH - 1, kH - 1), (kW - 1, kW - 1)),  # 全卷积需要的padding
                             mode='constant')

        # 执行转置卷积
        for i in range(H):
            for j in range(W):
                h_start = i
                h_end = h_start + kH
                w_start = j
                w_end = w_start + kW

                # 提取梯度局部区域 [batch, out_channels, kH, kW]
                grad_slice = grad_padded[:, :, h_start:h_end, w_start:w_end]

                # 计算输入梯度 [batch, in_channels]
                dX[:, :, i, j] = np.tensordot(
                    grad_slice,
                    W_rot180,
                    axes=([1, 2, 3], [0, 2, 3])  # 对齐维度：out_channels, kH, kW
                )

        # 保存梯度 ----------------------------------------------------------
        self.grads['W'] = dW + (self.weight_decay_lambda * self.W if self.weight_decay else 0)
        self.grads['b'] = db

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.y_true = None
        self.y_pred = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        # Softmax
        exp = np.exp(predicts - np.max(predicts, axis=1, keepdims=True))
        self.y_pred = exp / np.sum(exp, axis=1, keepdims=True)
        # 计算交叉熵损失
        batch_size = predicts.shape[0]
        loss = -np.log(self.y_pred[np.arange(batch_size), labels.astype(int)]).mean()
        self.y_true = labels
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        batch_size = self.y_true.shape[0]
        grads = self.y_pred.copy()
        grads[np.arange(batch_size), self.y_true] -= 1
        grads /= batch_size
        # Then send the grads to model for back propagation
        self.model.backward(grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.optimizable = False
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.mask = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        batch_size, channels, H, W = X.shape
        k = self.kernel_size
        stride = self.stride

        new_H = (H - k) // stride + 1
        new_W = (W - k) // stride + 1

        output = np.zeros((batch_size, channels, new_H, new_W))
        self.mask = np.zeros_like(X)

        for i in range(new_H):
            for j in range(new_W):
                h_start = i * stride
                h_end = h_start + k
                w_start = j * stride
                w_end = w_start + k

                X_slice = X[:, :, h_start:h_end, w_start:w_end]
                max_vals = np.max(X_slice, axis=(2, 3))
                output[:, :, i, j] = max_vals

                # 记录最大值位置
                mask = (X_slice == max_vals[:, :, None, None])
                self.mask[:, :, h_start:h_end, w_start:w_end] = mask

        return output

    def backward(self, grad):
        batch_size, channels, H, W = self.mask.shape
        k = self.kernel_size
        stride = self.stride
        new_H, new_W = grad.shape[2], grad.shape[3]

        dX = np.zeros_like(self.mask)
        for i in range(new_H):
            for j in range(new_W):
                h_start = i * stride
                h_end = h_start + k
                w_start = j * stride
                w_end = w_start + k

                dX[:, :, h_start:h_end, w_start:w_end] += grad[:, :, i, j][:, :, None, None] * self.mask[:, :,
                                                                                               h_start:h_end,
                                                                                               w_start:w_end]
        return dX


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False  # 标记该层无参数

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)