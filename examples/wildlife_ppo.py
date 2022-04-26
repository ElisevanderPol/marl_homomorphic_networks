from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.pg.mappo import MAPPO
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

from ops import get_agent_cls_wildlife


def build_and_train(env_id="WildlifeEnv-v0", run_ID=0, cuda_idx=None,
                    sample_mode="serial", n_parallel=2, args={}):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = SerialSampler  # (Ignores workers_cpus.)
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "cpu":
        Sampler = CpuSampler
        print(f"Using CPU parallel sampler (agent in workers), {gpu_cpu} for "
              "optimizing.")
    elif sample_mode == "gpu":
        Sampler = GpuSampler
        print(f"Using GPU parallel sampler (agent in master), {gpu_cpu} for "
              "sampling and optimizing.")
    elif sample_mode == "alternating":
        Sampler = AlternatingSampler
        affinity["workers_cpus"] += affinity["workers_cpus"]  # (Double list)
        affinity["alternating"] = True  # Sampler will check for this.
        print(f"Using Alternating GPU parallel sampler, {gpu_cpu} for "
              "sampling and optimizing.")

    env_kwargs = dict(id=env_id, n_agents=args.n_agents, w=args.grid_size,
                      h=args.grid_size)

    sampler = Sampler(
        EnvCls=gym_make,
        env_kwargs=env_kwargs,
        eval_env_kwargs=env_kwargs,
        batch_T=5,  # 5 time-steps per sampler iteration.
        batch_B=16,  # 16 parallel environments.
        max_decorrelation_steps=400,
        eval_n_envs=25,
        eval_max_steps=12500
    )

    algo = MAPPO(learning_rate=args.lr)

    agentCls, agent_basis = get_agent_cls_wildlife(args.network)

    agent = agentCls(model_kwargs={'basis': agent_basis,
                                   'channels': args.channels,
                                   'kernel_sizes': args.filters,
                                   'paddings': args.paddings,
                                   'fc_sizes': args.fcs,
                                   'strides': args.strides,
                                   'n_agents': args.n_agents})
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=args.n_steps,
        log_interval_steps=5e2,
        affinity=affinity,
    )


    config = dict(env_id=env_id, lr=args.lr,
                  debug=False,
                  network=args.network, fcs=str(args.fcs),
                  grid_size=args.grid_size, filters=args.filters,
                  n_agents=args.n_agents,
                  n_steps=args.n_steps, strides=str(args.strides),
                  channels=str(args.channels), paddings=str(args.paddings))

    str_fc = "_".join([str(x) for x in args.fcs])
    name = (f"{args.folder}_{args.network}_nagents={args.n_agents}_"
            f"gridsize={args.grid_size}_lr={args.lr}_"
            f"filters={str(args.filters[0])}_{args.n_steps}_l={str_fc}_")

    # Use same string for name and log_dir
    with logger_context(name, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder', help='Folder to store results',
                        default='WildlifeExperiment')
    parser.add_argument('--env_id', help='Gym', default='WildlifeEnv-v0')
    parser.add_argument('--lr', help='Learning rate', default=0.001, type=float)
    parser.add_argument('--n_agents', help='Number of agents', default=3,
                        type=int)
    parser.add_argument('--grid_size', help='height&width of grid',
                        default=21, type=int)
    parser.add_argument('--network', help='network type',
                        default='equivariant', type=str)
    parser.add_argument('--fcs', type=int, nargs='+', default=[256])
    parser.add_argument('--channels', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--filters',  type=int, nargs='+', default=[7, 5])
    parser.add_argument('--strides', type=int, nargs='+', default=[2, 1])
    parser.add_argument('--paddings', type=int, nargs='+', default=[0, 0])
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int,
                        default=0)
    parser.add_argument('--n_steps', type=int, default=3e5)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int,
                        default=None)
    parser.add_argument('--sample_mode', help='serial or parallel sampling',
                        type=str, default='serial',
                        choices=['serial', 'cpu', 'gpu', 'alternating'])
    parser.add_argument('--n_parallel', help='number of sampler workers',
                        type=int, default=2)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        sample_mode=args.sample_mode,
        n_parallel=args.n_parallel, args=args
    )
