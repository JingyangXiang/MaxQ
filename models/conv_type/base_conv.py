import torch
import torch.nn as nn


class BaseNMConv(nn.Module):
    def __init__(self):
        super(BaseNMConv, self).__init__()
        self.decay_start = 0
        self.decay_end = 0
        self.total_epoch = None
        self.cur_epoch = None
        self.M = None
        self.N = None
        self.schedule = None

    @torch.no_grad()
    def init_N_M_and_mask(self, N, M):
        # 初始化的mask根据权重的绝对值大小随机获取
        assert N <= M
        # for n:m sparsity
        self.N = N
        self.M = M
        weight = self.conv.weight
        weight_temp = weight.detach().abs().permute(0, 2, 3, 1).reshape(-1, self.M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(self.M - self.N)]
        mask = torch.ones(weight_temp.shape, device=weight_temp.device)
        mask = mask.scatter_(dim=1, index=index, value=0.).reshape(weight.permute(0, 2, 3, 1).shape)
        mask = mask.permute(0, 3, 1, 2)
        if hasattr(self, "mask"):
            self.mask.data = mask
        else:
            self.register_buffer('mask', mask)

    def init_prune_schedule(self, schedule: str):
        self.schedule = schedule
        print(f"init prune schedule: {schedule}")

    @torch.no_grad()
    def show_zero_num_and_update_mask(self):
        mask = self.get_mask(self.conv.weight)
        self.mask.data = mask
        print(f"weight num: {mask.numel()}, zero num: {torch.sum(torch.eq(mask, 0))}")

    @torch.no_grad()
    def do_grad_decay_v1(self, decay):
        raise NotImplementedError

    @torch.no_grad()
    def get_mask(self, weight):
        raise NotImplementedError

    @torch.no_grad()
    def update_epoch(self, cur_epoch, total_epoch, decay_start, decay_end):
        assert cur_epoch < total_epoch and total_epoch > 0
        assert 0 <= decay_start <= decay_end < total_epoch
        self.cur_epoch = cur_epoch
        self.total_epoch = total_epoch
        self.decay_start = decay_start
        self.decay_end = decay_end
        print(f"=> (cur_epoch, total_epoch, decay_start, decay_end) "
              f"= ({self.cur_epoch}, {self.total_epoch}, {self.decay_start}, {self.decay_end})")
