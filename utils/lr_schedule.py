import numpy as np
from params import LRScheduleParam


class LRSchedule():
    def __init__(self, param: LRScheduleParam) -> None:
        if param.lr_schedule == 'superconverge':
            def lr_schedule(t):
                return np.interp([t], [0, param.epochs * 2 // 5, param.epochs], [0, param.lr_max, 0])[0]
        elif param.lr_schedule == 'piecewise':
            def lr_schedule(t):
                if t / param.epochs < 0.5:
                    return param.lr_max
                elif t / param.epochs < 0.75:
                    return param.lr_max / 10.
                else:
                    return param.lr_max / 100.
        elif param.lr_schedule == 'linear':
            def lr_schedule(t):
                return np.interp([t], [0, param.epochs // 3, param.epochs * 2 // 3, param.epochs],
                                 [param.lr_max, param.lr_max, param.lr_max / 10, param.lr_max / 100])[0]
        elif param.lr_schedule == 'onedrop':
            def lr_schedule(t):
                if t < param.lr_drop_epoch:
                    return param.lr_max
                else:
                    return param.lr_one_drop
        elif param.lr_schedule == 'multipledecay':
            def lr_schedule(t):
                return param.lr_max - (t // (param.epochs // 10))*(param.lr_max / 10)
        elif param.lr_schedule == 'cosine':
            def lr_schedule(t):
                return param.lr_max * 0.5 * (1 + np.cos(t / param.epochs * np.pi))
        elif param.lr_schedule == 'cyclic':
            def lr_schedule(t):
                return np.interp([t], [0, 0.4 * param.epochs, param.epochs], [0, param.lr_max, 0])[0]

        elif param.lr_schedule == 'for_mart':
            # TODO deprecated this
            def lr_schedule(t):
                lr = param.lr_max
                if t >= 100:
                    lr = lr * 0.001
                elif t >= 90:
                    lr = lr * 0.01
                elif t >= 75:
                    lr = lr * 0.1
                return lr

        self.lr_schedule = lr_schedule

    def __call__(self, t):
        return self.lr_schedule(t)
