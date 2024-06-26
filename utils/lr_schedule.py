import numpy as np
import math
from params.lrscheduleparam import LRScheduleParam
from typing import List, Optional

# TODO Implement lr scheduler with torch.optim.lr_scheduler


class LRSchedule():
    def __init__(self,
                 lr_schedule: Optional[str] = None,
                 epochs: Optional[int] = None,
                 lr_max: Optional[float] = None,
                 epoch_list: Optional[List[int]] = None,
                 lr_list: Optional[List[float]] = None,
                 param: Optional[LRScheduleParam] = None,
                 ) -> None:
        if param is not None:
            self.lr_schedule = param.lr_schedule
            self.epochs = param.epochs
            self.lr_max = param.lr_max
            self.epoch_list = param.epoch_list[:]
            self.lr_list = param.lr_list[:]
        else:
            self.lr_schedule = lr_schedule
            self.epochs = epochs
            self.lr_max = lr_max
            self.epoch_list = epoch_list[:]
            self.lr_list = lr_list[:]

        if self.lr_schedule == "const":

            def lr_schedule_fn(t):
                return float(self.lr_max)

        elif self.lr_schedule == "stage":

            if len(self.epoch_list) == 0:
                self.epoch_list = [0]
            if len(self.lr_list) == 0:
                self.lr_list = [self.lr_max]
            if self.epoch_list[0] != 0:
                self.epoch_list = [0] + self.epoch_list
            if self.epoch_list[-1] != self.epochs:
                self.epoch_list = self.epoch_list + [self.epochs]
            for num1, num2 in zip(self.epoch_list[1:], self.epoch_list[:-1]):
                assert (num1 > num2)
            assert (len(self.epoch_list) == len(self.lr_list) + 1)

            def lr_schedule_fn(t):
                stage = sum([int(epoch <= t) for epoch in self.epoch_list]) - 1
                stage = min(max(0, stage), len(self.lr_list) - 1)
                return float(self.lr_list[stage])

        elif self.lr_schedule == "linear":

            if len(self.epoch_list) == 0:
                self.epoch_list = [0]
            if len(self.lr_list) == 0:
                self.lr_list = [self.lr_max]
            if self.epoch_list[0] != 0:
                self.epoch_list = [0] + self.epoch_list
            if self.epoch_list[-1] != self.epochs:
                self.epoch_list = self.epoch_list + [self.epochs]
            for num1, num2 in zip(self.epoch_list[1:], self.epoch_list[:-1]):
                assert (num1 > num2)
            assert (len(self.epoch_list) == len(self.lr_list))

            def lr_schedule_fn(t):
                return float(np.interp(t, self.epoch_list, self.lr_list))

        elif self.lr_schedule == "cosine":

            assert len(self.epoch_list) == 1
            assert self.epoch_list[0] >= 0

            def lr_schedule_fn(t):
                if t < self.epoch_list[0]:
                    factor = (t + 0.5) / self.epoch_list[0]
                    lr = self.lr_max * factor
                else:
                    factor = (t - self.epoch_list[0]) / (self.epochs - self.epoch_list[0])
                    lr = 1e-6 + (self.lr_max - 1e-6) * 0.5 * (1 + math.cos(factor * math.pi))
                return lr

        else:

            raise ValueError("No such a type")

        self.lr_schedule_fn = lr_schedule_fn

    def __call__(self, t):

        return self.lr_schedule_fn(t)

    def stage(self, t):

        assert (self.lr_schedule == "stage")

        return sum([int(epoch <= t) for epoch in self.epoch_list]) - 1
