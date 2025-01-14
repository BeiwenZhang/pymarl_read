import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

# 注册表，用于存储不同的动作选择策略
REGISTRY = {}

# 多项式分布动作选择器类
class MultinomialActionSelector():

    def __init__(self, args):
        # 初始化类，接受参数
        self.args = args

        # 设置epsilon衰减策略
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        # 初始化epsilon值
        self.epsilon = self.schedule.eval(0)
        # 是否在测试模式下使用贪婪策略
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # 选择动作的方法

        # 克隆输入数据并对不可用的动作进行掩码
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        # 更新epsilon值
        self.epsilon = self.schedule.eval(t_env)

        # 测试模式下使用贪婪策略选择动作，否则使用多项式分布选择动作
        if test_mode and self.test_greedy:
            # 贪婪策略：选择概率最高的动作
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            # 否则根据概率分布随机选择动作
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


# 将多项式选择策略注册到REGISTRY中
REGISTRY["multinomial"] = MultinomialActionSelector


# Epsilon-greedy动作选择器类
class EpsilonGreedyActionSelector():

    def __init__(self, args):
        # 初始化类，接受参数
        self.args = args

        # 设置epsilon衰减策略
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        # 初始化epsilon值
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # 选择动作的方法

        # 假设agent_inputs是每个智能体的Q值（Q-Values）
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # 测试模式下，只执行贪婪选择，不使用epsilon
            self.epsilon = 0.0

        # 对不可用的动作进行掩码处理
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # 不可选的动作Q值设置为负无穷，确保它们不会被选择

        # 生成随机数并根据epsilon选择是否执行随机动作
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()  # 如果小于epsilon则选择随机动作
        random_actions = Categorical(avail_actions.float()).sample().long()  # 从可用动作中随机选择

        # 根据epsilon的值，选择随机动作或使用最大Q值的贪婪策略
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        
        return picked_actions


# 将epsilon-greedy选择策略注册到REGISTRY中
REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
