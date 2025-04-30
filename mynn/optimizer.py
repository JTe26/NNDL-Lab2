from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]

class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}
        for layer in self.model.layers:
            if layer.optimizable:  # 仅处理可优化层（如 Conv2D、Linear）
                for key in layer.params:
                    self.velocity[f"{id(layer)}.{key}"] = np.zeros_like(layer.params[key])

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params:
                    # 更新参数
                    v_key = f"{id(layer)}.{key}"
                    self.velocity[v_key] = self.mu * self.velocity[v_key] + self.init_lr * layer.grads[key]
                    layer.params[key] -= self.velocity[v_key]

                    # 仅在启用 L2 正则化时应用权重衰减
                    if hasattr(layer, 'weight_decay') and layer.weight_decay:
                        layer.params[key] -= self.init_lr * layer.weight_decay_lambda * layer.params[key]
                layer.clear_grad()  # 添加梯度清零