import argparse
import logging
from bayesian_experimental_design import BED, param_search
from cheater import Cheater
from agents import RLAgent


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for QMR model')
    # Data path arguments
    parser.add_argument('--input_dir', type=str, default='dataset')
    parser.add_argument('--output_dir', type=str, default='output')

    # Data set arguments
    parser.add_argument('--dataset_name', type=str, default='SymCAT')
    parser.add_argument('--n_diseases', type=int, default=200)

    # Common arguments
    parser.add_argument('--solver', type=str, required=True)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--max_episode_len', type=int, default=15)

    # Environment arguments
    parser.add_argument('--no_single_diag_action', action='store_true')
    parser.add_argument('--no_mask_actions', action='store_true')
    parser.add_argument('--aux_reward', action='store_true')
    # parser.add_argument('--diag_aux', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--lambda_', type=float, default=0.1)

    # RL agent arguments
    parser.add_argument('--algorithm', type=str, default='PPO')
    parser.add_argument('--training_iteration', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--checkpoint_path', type=str)
    # parser.add_argument('--resume', action='store_true')

    # Bayesian experimental design arguments
    parser.add_argument('--utility_func', type=str, default='KL',
                        help="Chose from (`SI` and `KL`), case insensitive.")
    parser.add_argument('--threshold', type=float, default='0.05')
    parser.add_argument('--param_search', action='store_true')

    # Cheater arguments
    parser.add_argument('--cheater_method', type=str, default='inference')

    args = parser.parse_args()
    args.single_diag_action = False if args.no_single_diag_action else True
    args.mask_actions = False if args.no_mask_actions else True
    return args


def main():
    args = parse_arguments()

    if args.solver == 'bed' or args.solver == 'cheater':
        logging.basicConfig(level=logging.INFO)
        args.single_diag_action = True
        args.mask_actions = False
        args.aux_reward = False

    if args.solver == 'bed' and args.param_search:
        param_search(args)
        return

    if args.solver.lower() == 'bed':
        solver = BED(args)
    elif args.solver.lower() == 'cheater':
        solver = Cheater(args)
    elif args.solver.lower() == 'rl':
        solver = RLAgent(args)
    else:
        raise NotImplementedError

    solver.run()


if __name__ == '__main__':
    main()
