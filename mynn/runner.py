import pickle
import numpy as np
import os
from tqdm import tqdm
import time
from scipy.ndimage import rotate, shift, zoom

class RunnerM():

    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        # 训练记录
        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.best_score = 0

    def train(self, train_set, dev_set, **kwargs):
        # 参数设置
        self.model.training = True
        num_epochs = kwargs.get("num_epochs", 5)
        eval_interval = kwargs.get("eval_interval", 1000)  # 评估间隔
        save_dir = kwargs.get("save_dir", "best_model")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 训练循环
        for epoch in range(num_epochs):
            X, y = train_set
            X, y = self._shuffle_data(X, y)

            # 使用tqdm进度条
            num_batches = int(np.ceil(len(X) / self.batch_size))
            pbar = tqdm(range(num_batches), desc=f'Epoch {epoch + 1}/{num_epochs}')

            for iteration in pbar:
                # 获取当前批次
                batch_X, batch_y = self._get_batch(X, y, iteration)
                # 前向传播
                logits = self.model(batch_X)

                trn_loss = self.loss_fn(logits, batch_y)
                trn_score = self.metric(logits, batch_y)

                # 记录训练指标
                self.train_loss.append(trn_loss)
                self.train_scores.append(trn_score)

                # 在进度条显示当前训练指标
                pbar.set_postfix({
                    'train_loss': f'{trn_loss:.4f}',
                    'train_acc': f'{trn_score:.4f}'
                })

                # 反向传播和优化
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                # 每eval_interval次迭代评估一次验证集
                if (iteration + 1) % eval_interval == 0 or iteration == num_batches - 1:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)

                    # 更新最佳模型
                    if dev_score > self.best_score:
                        self.model.save_model(os.path.join(save_dir, 'MLP_l2_2.pickle'))
                        pbar.write(f'best accuracy performance has been updated: {self.best_score:.5f} --> {dev_score:.5f}"')
                        self.best_score = dev_score

        print(f"Training Complete! Best Dev Accuracy: {self.best_score:.5f}")


    def evaluate(self, data_set, batch_size=None):
        """批量评估模型性能，仅使用numpy库，并输出每个阶段的时间"""
        X, y = data_set


        # 确保X和y是numpy数组
        X = np.array(X)
        y = np.array(y)

        # 测量模型推理时间

        logits = self.model(X)

        # 测量损失计算时间

        total_loss = self.loss_fn(logits, y)

        total_score = self.metric(logits, y)


        # 计算平均损失和得分
        return total_score,total_loss


    def _shuffle_data(self, X, y):
        """打乱数据"""
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    def _get_batch(self, X, y, iteration):
        """获取批次数据"""
        start = iteration * self.batch_size
        end = (iteration + 1) * self.batch_size
        return X[start:end], y[start:end]