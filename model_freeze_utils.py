import torch
import torch.nn as nn

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def set_trainable(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def set_norm_trainable(model, flag: bool):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm)):
            m.train(flag)
            for p in m.parameters():
                p.requires_grad = flag

# Unfreeze a specific stage of the model
def unfreeze_stage(model, stage_idx: int):
    set_trainable(model.stages[stage_idx], True)

# Define the unfreezing groups, alongside changes to learning rates
def llrd_groups(model, base_lr=3e-5, wd=1e-4):
    order = [
        ('head', model.head, 1.00),
        ('s3',   model.stages[3], 0.75),
        ('s2',   model.stages[2], 0.56),
        ('s1',   model.stages[1], 0.42),
        ('s0',   model.stages[0], 0.31),
        ('stem', model.stem,       0.23),
    ]
    groups = []
    for _, mod, m in order:
        params = [p for p in mod.parameters() if p.requires_grad]
        if params:
            groups.append({'params': params, 'lr': base_lr*m, 'weight_decay': wd})
    return groups

# Define the warmup scheduler
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, after_scheduler=None):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    # Calculate and return the LR for a specific stage
    def get_lr(self):
        if not self.finished:
            return [base_lr * self.last_epoch / float(self.warmup_steps)
                    for base_lr in self.base_lrs]
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_steps:
            super().step(epoch)
        else:
            if not self.finished:
                self.after_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                self.finished = True
            if self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup_steps)