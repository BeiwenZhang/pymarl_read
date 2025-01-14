import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

#注意 在sacred中这个main 函数是sacred框架的入口点，它定义了框架的配置、观察者、运行函数等。
@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

#帮我写中文注释
#帮我写中文注释
if __name__ == '__main__':
    # 深拷贝系统参数以避免修改原始参数
    params = deepcopy(sys.argv)

    # 获取默认配置，从default.yaml文件中读取配置信息
    # 使用os.path.join确保代码兼容不同操作系统路径格式
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            # 尝试加载并解析yaml文件
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            # 如果yaml文件解析失败，抛出异常并提供错误信息
            assert False, "default.yaml error: {}".format(exc)

    # 加载算法和环境的基础配置
    # 从命令行参数中获取环境配置，如果没有指定，则使用默认配置
    #假设在命令行中使用了 --env-config=sc2，那么 _get_config 就会从 src/config/envs 目录中加载 sc2 相关的环境配置文件。
    env_config = _get_config(params, "--env-config", "envs")
    # 从命令行参数中获取算法配置，如果没有指定，则使用默认配置
    alg_config = _get_config(params, "--config", "algs")
    # 更新配置字典，整合默认配置、环境配置和算法配置
    # config_dict = {**config_dict, **env_config, **alg_config}
    # 使用recursive_dict_update函数而不是简单的字典解包，以处理嵌套配置
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # 将所有配置添加到sacred实验中
    ex.add_config(config_dict)

    # 默认情况下，将结果保存到磁盘上，为sacred配置文件存储观察者
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # 使用命令行参数运行sacred实验
    ex.run_commandline(params)

