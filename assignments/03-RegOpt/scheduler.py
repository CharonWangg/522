from typing import List
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_iters, min_lr=0.0, last_epoch=-1):
        """
        Cosine learning rate scheduler.

        Arguments:
            optimizer (torch.optim.Optimizer): The optimizer to use.
            last_epoch (int): The last epoch.
            min_lr (float): The minimum learning rate.
        """
        # construct a cosine learning rate scheduler
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # write a cosine learning rate scheduler
        current_lrs = []
        for base_lr in self.base_lrs:
            current_lr = (
                self.min_lr
                + (base_lr - self.min_lr)
                * (1 + np.cos(np.pi * self.last_epoch / self.max_iters))
                / 2
            )
            current_lrs.append(current_lr)
        return current_lrs
