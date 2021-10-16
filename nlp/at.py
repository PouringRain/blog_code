import torch
import torch.nn as nn

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# 初始化
fgm = FGM(model)
for batch_input, batch_label in data:
  # 正常训练
  loss = model(batch_input, batch_label)
  loss.backward()
  # 对抗训练
  fgm.attack() # 修改embedding
  # optimizer.zero_grad() # 梯度累加，不累加去掉注释
  loss_sum = model(batch_input, batch_label)
  loss_sum.backward() # 累加对抗训练的梯度
  fgm.restore() # 恢复Embedding的参数

  optimizer.step()
  optimizer.zero_grad()
