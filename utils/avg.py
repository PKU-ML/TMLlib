import math


class AverageMeter():

    def __init__(self, name: str = 'No name'):
        self.name: str = name
        self.sum: float = 0.0
        self.num: int = 0
        self.mean: float = 0.0
        self.now: float = 0.0

    def reset(self):
        self.sum = 0.0
        self.num = 0
        self.mean = 0.0
        self.now = 0.0

    def update(self, mean_var: float, count: int = 1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')

        self.sum += mean_var * count
        self.num += count
        self.mean = float(self.sum) / self.num
        self.now = mean_var
