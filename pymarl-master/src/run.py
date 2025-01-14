import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

def run(_run, _config, _log):
    """
    执行实验的主要函数。

    参数:
    - _run: Sacred的run对象，用于跟踪实验。
    - _config: 包含实验参数的字典。
    - _log: 日志对象，用于记录实验信息。

    返回值:
    无
    """
    # 检查配置参数的有效性
    _config = args_sanity_check(_config, _log)

    # 使用配置参数初始化一个SimpleNamespace对象，以便通过点符号访问参数
    args = SN(**_config)
    # 根据use_cuda参数决定使用CUDA还是CPU
    args.device = "cuda" if args.use_cuda else "cpu"

    # 设置日志记录器
    logger = Logger(_log)

    # 记录实验参数
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # 配置TensorBoard日志记录器
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # 默认启用Sacred日志记录
    logger.setup_sacred(_run)

    # 执行训练
    run_sequential(args=args, logger=logger)

    # 实验结束后清理
    print("Exiting Main")

    # 停止所有线程
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    # 退出脚本
    print("Exiting script")

    # 确保程序真的退出
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()
def run_sequential(args, logger):
    """
    顺序运行训练或测试流程。

    参数:
    - args: 命令行参数或其他配置参数。
    - logger: 用于记录日志信息的对象。

    该函数负责初始化运行器、设置环境信息、配置学习器，并根据命令行参数
    决定是进行训练还是测试。在训练过程中，它还负责定期保存模型和执行测试运行。
    """

    # 初始化运行器以获取环境信息
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # 定义基础方案
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # 创建回放缓冲区
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # 设置多智能体控制器
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 初始化学习器
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    # 加载检查点，如果指定的话
    if args.checkpoint_path != "":
 # 加载模型并设置训练或测试模式
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        #智能体学习器加载checkpoint
        learner.load_models(model_path)
        runner.t_env = timestep_to_load
#如果是评测模式 或者保存回放
        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

      # 开始训练
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # 运行一整集并进行训练
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        # 当缓冲区可以提供样本时，进行训练
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # 将样本截断到仅包含填充的时间步
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            # 如果样本设备与指定设备不同，则转移到指定设备
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # 使用样本进行学习
            learner.train(episode_sample, runner.t_env, episode)

        # 定期执行测试运行
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            # 打印当前训练进度和时间信息
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            # 更新最后一次测试时间
            last_test_T = runner.t_env
            # 执行多个测试运行
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # 定期保存模型
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            # 准备模型保存路径
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # 保存模型
            learner.save_models(save_path)

        # 更新当前集数
        episode += args.batch_size_run

        # 定期记录日志
        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # 关闭环境并结束训练
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
