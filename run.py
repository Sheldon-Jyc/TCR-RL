import argparse
import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import sys
sys.path.append('...')
# from gops
from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="elfin_mujoco", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", default=829, help="Global seed")
    parser.add_argument("--reward_scale", type=float, default=0.2, help="reward scale factor")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValueDistri",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", )
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[256,256])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="relu",
    )
    parser.add_argument("--value_output_activation", type=str, default="linear",)
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="MLP",
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_sensory_units", type=int, default=20)
    parser.add_argument("--policy_command_units", type=int, default=6)
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256,512,2048,2048,2046,512,256])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="gelu",
    )
    parser.add_argument("--policy_std_type", type=str, default="parameter")
    parser.add_argument("--policy_min_log_std", type=int, default=-1)
    parser.add_argument("--policy_max_log_std", type=int, default=0.8)
    parser.add_argument("--value_learning_rate", type=float, default=0.0001)
    parser.add_argument("--policy_learning_rate", type=float, default=0.0001)
    parser.add_argument("--alpha_learning_rate", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--delay_update", type=int, default=2)
    parser.add_argument("--max_iteration", type=int, default=2000000)
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer",
    )
    parser.add_argument("--buffer_warm_size", type=int, default=10000)
    parser.add_argument("--buffer_max_size", type=int, default=2*500000)
    parser.add_argument("--replay_batch_size", type=int, default=256)
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
    parser.add_argument("--sample_batch_size", type=int, default=20)
    parser.add_argument("--noise_params", type=dict, default=None)
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--apprfunc_save_interval", type=int, default=10000)
    parser.add_argument("--log_save_interval", type=int, default=2000)
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)
    alg = create_alg(**args)
    sampler = create_sampler(**args)
    buffer = create_buffer(**args)
    evaluator = create_evaluator(**args)
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)
    trainer.train()
    print("Training is finished!")
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
