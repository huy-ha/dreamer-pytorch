import datetime
import os
import argparse
import torch

from rlpyt.samplers.collections import TrajInfo
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.context import logger_context

from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.algos.dreamer_dbc import DreamerDBC
from dreamer.envs.dmc import DeepMindControl
from dreamer.envs.time_limit import TimeLimit
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from dreamer.envs.wrapper import make_wapper, make_wapper_custom
from copy import deepcopy
import dmc2gym
import pickle
import random
import numpy as np
from tqdm import tqdm


def seed_all(seed):
    print(f"SEEDING WITH {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_and_eval(output_pickle: str,
                   domain_name: str,
                   task_name: str,
                   eval_resource_files: str,
                   img_source: str,
                   total_frames: int,
                   eval_steps: int,
                   seed: int,
                   image_size: int,
                   cuda_idx=None, eval=False,
                   load_model_path=None, action_repeat=2):
    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')
    seed_all(args.seed)

    factory_method = make_wapper_custom(
        wrapper_classes=[ActionRepeat, NormalizeActions, TimeLimit],
        wrapper_kwargs=[dict(amount=action_repeat), dict(), dict(duration=1000 / action_repeat)])
    env_kwargs = dict(
        domain_name=domain_name,
        task_name=task_name,
        resource_files=eval_resource_files,
        img_source=img_source,
        total_frames=total_frames,
        seed=seed,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat)
    agent = DMCDreamerAgent(
        train_noise=0.0,
        eval_noise=0,
        expl_type="additive_gaussian",
        expl_min=None,
        expl_decay=None,
        initial_model_state_dict=agent_state_dict)
    env = factory_method(**env_kwargs)
    agent.initialize(env.spaces, share_memory=False,
                     global_B=1, env_ranks=[0])
    env.seed(args.seed)
    count = 0
    observation = env.reset()
    reward = torch.tensor(0.0)
    action = torch.tensor(0.0)
    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []
    next_observation = []
    agent_infos = []
    infos = []
    with tqdm(total=eval_steps, dynamic_ncols=True, desc='Collecting eval experience') as pbar:
        while count < eval_steps:
            observation = torch.tensor(observation)
            agent_step = agent.step(
                observation=observation,
                prev_action=action,
                prev_reward=reward)
            action = agent_step.action
            next_observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)
            infos.append(info)
            agent_infos.append(agent_step.agent_info)
            prev_action = action
            observation = next_observation
            count += 1
            pbar.update(1)
            if done:
                observation = env.reset()
                reward = torch.tensor(0.0)
                action = torch.tensor(0.0)
    pickle.dump(
        {
            'obs': observations,
            'action': actions,
            'next_obs': next_observations,
            'reward': rewards,
            'done': dones,
            'info': infos,
            'agent_infos': agent_infos
        },
        open(output_pickle, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_pickle', default='experience.pkl')
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='balance')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--eval_resource_files', type=str, required=True)
    parser.add_argument('--img_source', default='video', type=str,
                        choices=['color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--total_frames', default=25000, type=int)
    parser.add_argument('--eval_steps', default=100000, type=int)
    parser.add_argument('--cuda-idx',
                        help='gpu to use ',
                        type=int,
                        default=None)
    # path to params.pkl
    parser.add_argument('--load-model-path',
                        help='load model from path',
                        type=str,
                        required=True)
    args = parser.parse_args()
    build_and_eval(**vars(args))
