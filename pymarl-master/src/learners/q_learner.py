import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop

#mixer 代表了一个混合器（Mixer）对象，用于组合来自各个智能体的 Q 值，以便生成一个全局的 Q 值，用于进一步的强化学习更新
class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1


# 训练过程 (train):

# 获取批次数据：从 batch 中获取奖励、动作、状态、可用动作等信息。
# 计算当前 Q 值：通过 mac（多智能体控制器）来获得每个智能体的 Q 值预测。
# 计算目标 Q 值：通过 target_mac 来计算目标网络的 Q 值，并根据策略（如双重 Q 学习）来选择目标 Q 值。
# 混合 Q 值：如果 mixer 存在，则使用它来混合当前的 Q 值和目标 Q 值。具体来说，mixer 会根据每个智能体的 Q 值和全局状态来生成一个全局的 Q 值。
# 计算损失和优化：计算当前 Q 值和目标 Q 值之间的 TD 错误，然后通过优化器更新参数。
# 目标网络更新：根据设定的更新间隔（target_update_interval）来更新目标网络和目标混合器。
#更新参数的过程就是学习的过程
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        训练模型。
        
        参数:
        - batch: EpisodeBatch类型，包含一个批次的episode数据。
        - t_env: int类型，当前环境的步数，用于日志记录和更新判断。
        - episode_num: int类型，当前episode的编号，用于判断是否更新目标网络。
        
        该方法没有返回值。
        """

        # 获取相关数据
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # 计算估计的Q值
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # 按时间连接

        # 选择每个智能体采取的动作的Q值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # 移除最后一个维度

        # 计算目标所需的Q值
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # 我们不需要第一个时间步的Q值估计来计算目标
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # 按时间连接

        # 掩码掉不可用的动作
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # 目标Q值的最大值
        if self.args.double_q:
            # 获取最大化实时Q的动作（用于双重Q学习）
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # 混合
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # 计算1步Q学习目标
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # TD误差
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 将来自填充数据的目标置为0
        masked_td_error = td_error * mask

        # 标准L2损失，仅在实际数据上取平均
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # 优化
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # 更新目标网络
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # 日志记录
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
