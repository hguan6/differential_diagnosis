import logging
import numpy as np
import random
from multiprocessing import Pool
import itertools

from QMR import QMR

logger = logging.getLogger(__name__)


class BED:
    def __init__(self, args):
        self.qmr = QMR({'args': args})
        self.args = args
        self.max_episode_len = args.max_episode_len
        self.threshold = args.threshold
        self.util_func = self.utility_func(args.utility_func)

    def utility_func(self, func_name):
        def eps(prob):
            return prob + np.finfo(np.float).eps

        def kl_divergence(prob1, prob2):
            return (np.log(eps(prob1)) * prob1 - np.log(eps(prob2)) * prob1).sum()

        def shannon_information(prob1, prob2):
            return (np.log(eps(prob1)) * prob1 - np.log(eps(prob2)) * prob2).sum()

        if func_name.lower() == 'kl':
            return kl_divergence
        elif func_name.lower() == 'si':
            return shannon_information
        else:
            raise NotImplementedError(
                f'Utility function {func_name} is not defined. Must be one of (`KL`, `SI`).')

    def act(self):
        candidate_findings = self.qmr.get_candidate_finding_index()
        visited_findings = np.argwhere(self.qmr.obs != 0).ravel()
        pos_findings = np.argwhere(self.qmr.obs == 1).ravel().tolist()
        neg_findings = np.argwhere(self.qmr.obs == -1).ravel().tolist()

        old_probs, old_joint = self.qmr.compute_disease_probs(
            pos_findings, neg_findings, normalize=True)
        max_utility = 0
        action = None
        for finding in candidate_findings:
            if finding not in visited_findings:
                # When the inquired finding is positive
                new_probs, new_joint = self.qmr.compute_disease_probs(
                    pos_findings + [finding], neg_findings, normalize=True)
                p_pos = new_joint / old_joint
                utility = p_pos * self.util_func(new_probs, old_probs)
                # When the inquired finding is negative
                new_probs, new_joint = self.qmr.compute_disease_probs(
                    pos_findings, neg_findings + [finding], normalize=True)
                p_neg = new_joint / old_joint
                utility += p_neg * self.util_func(new_probs, old_probs)
                if utility > max_utility:
                    max_utility = utility
                    action = finding

        if max_utility < self.threshold or action is None:
            action = self.qmr.n_all_findings
        return action

    def run(self):
        n_correct = [0, 0, 0]
        total_steps = 0
        if self.qmr.test_data is None:
            test_size = self.args.test_size
        else:
            test_size = len(self.qmr.test_data)

        for i in range(test_size):
            if self.qmr.test_data is None:
                self.qmr.reset()
            else:
                self.qmr.reset(i)
            for step in range(self.max_episode_len):
                action = self.act()

                if action == self.qmr.n_all_findings:
                    break
                self.qmr.step(action)
                # _, _, done, _ = self.qmr.step(action)
                # if done:
                #     break
            correctness = self.qmr.inference()
            n_correct = [i + j for i, j in zip(n_correct, correctness)]
            total_steps += step + 1
        accuracy = [i / test_size for i in n_correct]
        print(
            f'max_episode_len: {self.max_episode_len}, threshold: {self.threshold}\n#experiments: {test_size}; accuracy: {accuracy}; average steps: {total_steps/test_size:.4f}')


def job(pack):
    args, max_episode_len, threshold = pack
    bed = BED(args)
    bed.max_episode_len = max_episode_len
    bed.threshold = threshold
    bed.run()


def param_search(args):
    pack = [(args, l, t)
            for l, t in itertools.product((20, 15, 10), (0.01, 0.05, 0.1))]
    with Pool() as p:
        p.map(job, pack)
