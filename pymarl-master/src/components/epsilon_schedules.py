import numpy as np


# 衰减然后保持平稳的调度类，用于实现 epsilon 的变化（例如在 epsilon-greedy 策略中使用）
class DecayThenFlatSchedule():

    def __init__(self,
                 start,  # 初始值，衰减开始时的值
                 finish,  # 最终值，衰减结束后保持的值
                 time_length,  # 衰减的持续时间
                 decay="exp"):  # 衰减类型（默认为指数衰减），支持 "exp" 或 "linear" 衰减

        # 保存初始化参数
        self.start = start
        self.finish = finish
        self.time_length = time_length
        
        # 计算线性衰减的步长
        self.delta = (self.start - self.finish) / self.time_length
        
        # 设置衰减方式（默认使用指数衰减）
        self.decay = decay

        # 如果选择的是指数衰减，计算衰减的缩放因子
        if self.decay in ["exp"]:
            # 通过 np.log 计算缩放因子，确保最终值能够接近 finish
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        """
        计算在时间步骤 T 下的 epsilon 值。
        :param T: 当前的时间步
        :return: 计算得到的 epsilon 值
        """
        if self.decay in ["linear"]:
            # 对于线性衰减，直接使用线性公式计算epsilon
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            # 对于指数衰减，使用指数函数计算epsilon
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass

 