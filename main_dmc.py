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


def build_and_train(log_dir, run_ID=0, cuda_idx=None, eval=False,
                    load_model_path=None, args=None, action_repeat=2):
    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')

    factory_method = make_wapper_custom(
        wrapper_classes=[ActionRepeat, NormalizeActions, TimeLimit],
        wrapper_kwargs=[dict(amount=action_repeat), dict(), dict(duration=1000 / action_repeat)])
    env_kwargs = dict(
        domain_name=args.domain_name,
        task_name=args.task_name,
        resource_files=args.resource_files,
        img_source=args.img_source,
        total_frames=args.total_frames,
        seed=args.seed,  # TODO seed other parts
        visualize_reward=False,
        from_pixels=True,
        height=args.image_size,
        width=args.image_size,
        frame_skip=action_repeat)
    eval_env_kwargs = deepcopy(env_kwargs)
    eval_env_kwargs.update({
        "resource_files": args.eval_resource_files
    })
    sampler = SerialSampler(
        EnvCls=factory_method,
        TrajInfoCls=TrajInfo,
        env_kwargs=env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )

    # Run with defaults.
    algo = Dreamer(initial_optim_state_dict=optimizer_state_dict)\
        if args.approach == 'dreamer' else \
        DreamerDBC(
        bisim_coef=args.bisim_coeff,
        initial_optim_state_dict=optimizer_state_dict)
    agent = DMCDreamerAgent(
        train_noise=0.3,
        eval_noise=0,
        expl_type="additive_gaussian",
        expl_min=None,
        expl_decay=None,
        initial_model_state_dict=agent_state_dict)
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        seed=args.seed,
        n_steps=5e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    game = f'{args.domain_name}_{args.task_name}'
    config = dict(game=game)
    name = "dreamer_" + game
    with logger_context(
            log_dir, run_ID, name, config,
            snapshot_mode='gap', override_prefix=True,
            use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--approach',
                        default='ours',
                        choices=['ours', 'dreamer'])
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='balance')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--resource_files', type=str, required=True)
    parser.add_argument('--eval_resource_files', type=str, required=True)
    parser.add_argument('--img_source', default='video', type=str,
                        choices=['color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--bisim_coeff', default=1.0, type=float)
    parser.add_argument('--total_frames', default=25000, type=int)
    parser.add_argument('--run-ID',
                        help='run identifier (logging)',
                        type=int,
                        default=0)
    parser.add_argument('--cuda-idx',
                        help='gpu to use ',
                        type=int,
                        default=None)
    parser.add_argument('--eval', action='store_true')
    # path to params.pkl
    parser.add_argument('--load-model-path',
                        help='load model from path',
                        type=str)
    default_log_dir = os.path.join(
        os.path.dirname(__file__),
        'logs',
        datetime.datetime.now().strftime("%Y%m%d"))
    parser.add_argument('--log-dir', type=str, default=default_log_dir)
    args = parser.parse_args()
    log_dir = os.path.abspath(args.log_dir)
    i = args.run_ID
    while os.path.exists(os.path.join(log_dir, 'run_' + str(i))):
        print(f'run {i} already exists. ')
        i += 1
    print(f'Using run id = {i}')
    args.run_ID = i
    build_and_train(
        log_dir,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        eval=args.eval,
        load_model_path=args.load_model_path,
        args=args)
