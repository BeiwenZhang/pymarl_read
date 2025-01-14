from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
#也就是多个agent（RNN）的控制器，相当于单智能体和环境交互的接口，变成一个多智能体环境
class BasicMAC:
    #mac=multi agent controller
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """
        根据给定的episode批次、episode时间步和环境时间步来选择动作。
        
        参数:
        - ep_batch: 一个包含episode数据的批次。
        - t_ep: 当前的episode时间步。
        - t_env: 当前的环境时间步。
        - bs: 切片对象，用于选择批次中的元素，默认为全部元素。
        - test_mode: 布尔值，表示是否在测试模式下运行。
        
        返回:
        - chosen_actions: 选择的动作。
        
        此函数的主要逻辑是根据智能体的输出和可用的动作来选择最合适的动作。它首先获取当前时间步下所有可用的动作，
        然后通过调用智能体的forward方法获取智能体的输出，最后使用动作选择器来选择具体的动作。
        """
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
#给我生成中文注释！！
    def forward(self, ep_batch, t, test_mode=False):
        """
        前向计算过程，处理输入并生成策略输出。
        
        参数:
        - ep_batch: 一个批次的回合数据，包含代理的观测、可用动作等信息。
        - t: 当前时间步。
        - test_mode: 是否为测试模式，影响策略的输出（默认为False）。
        
        返回:
        - 经过处理的策略输出，形状为(batch_size, n_agents, -1)。
        """
        # 构建代理输入数据
        agent_inputs = self._build_inputs(ep_batch, t)
        # 获取当前时间步所有代理的可用动作
        avail_actions = ep_batch["avail_actions"][:, t]
        # 通过代理模型进行前向计算，得到原始策略输出和隐藏状态
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # 如果代理的输出类型是策略对数概率，则进行softmax处理
        if self.agent_output_type == "pi_logits":
            # 根据参数设置，判断是否在softmax前屏蔽不可用动作
            if getattr(self.args, "mask_before_softmax", True):
                # 将不可用动作的对数概率设为非常小的值，以减小其对softmax的影响
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            # 对代理输出应用softmax，转换为概率分布
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # 在非测试模式下，应用epsilon贪婪策略
                # 计算epsilon floor，即以一定概率选择均匀分布的动作
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # 如果设置了在softmax前屏蔽不可用动作，则计算可用动作的数量
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                # 应用epsilon贪婪策略，以探索和利用的平衡
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                            + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # 再次屏蔽不可用动作的概率
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        # 返回处理后的策略输出（概率分布）
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
