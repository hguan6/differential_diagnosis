import json
import random
from collections import Counter
import itertools
from copy import deepcopy

import numpy as np
from gym import spaces, Env
from gym.utils import seeding

from data_utils import load_data


class QMR(Env):
    def __init__(self, config):
        # Set arguments
        self.args = config['args']
        self.single_diag_action = self.args.single_diag_action
        self.aux_reward = self.args.aux_reward
        # self.diag_aux = self.args.diag_aux
        self.gamma = self.args.gamma
        self.lambda_ = self.args.lambda_
        self.max_episode_len = self.args.max_episode_len

        # Set random seed
        self.set_seed()

        # Load dataset
        data_info = load_data(self.args)
        self.finding2disease, self.disease2finding, self.p_d, self.test_data = data_info
        self.n_all_findings = len(self.finding2disease)
        self.n_all_diseases = len(self.disease2finding)
        # print(
        #     f"Number of diseases: {self.n_all_diseases}, Number of findings: {self.n_all_findings}")
        # avg_diseases = sum(
        #     [len(v) for v in self.finding2disease.values()]) / self.n_all_findings
        # print(f'average #diseases for each finding: {avg_diseases}')
        # avg_findings = sum(
        #     [len(v) for v in self.disease2finding.values()]) / self.n_all_diseases
        # print(f'average #findings for each disease: {avg_findings}')

        # Define action space and observation space
        self.action_size = self.n_all_findings
        self.action_size += 1 if self.args.single_diag_action else self.n_all_diseases
        self.action_space = spaces.Discrete(self.action_size)

        obs_space = spaces.Box(-1, 1, shape=(
            self.n_all_findings,))
        self.mask_actions = self.args.mask_actions
        if self.mask_actions:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.action_size,)),
                # "avail_actions": spaces.Box(0, 1, shape=(self.action_size,)),
                "obs": obs_space
            })
        else:
            self.observation_space = obs_space

        self.visited = {}

    def one_disease_sample(self, floor_prob=0.0):
        findings = set()
        while True:
            disease = random.randrange(self.n_all_diseases)
            for finding, prob in self.disease2finding[disease].items():
                if random.random() <= prob:
                    findings.add(finding)
            if floor_prob > 0:
                for f in range(self.n_all_findings):
                    if random.random() < floor_prob:
                        findings.add(f)
            if len(findings) != 0:
                break
        findings = tuple(findings)
        first_finding = random.choice(findings)
        return disease, findings, first_finding

    def get_candidate_finding_index(self):
        pos_findings = np.argwhere(self.obs == 1).ravel()
        pos_bytes = pos_findings.tobytes()
        if pos_bytes in self.visited:
            return self.visited[pos_bytes]
        else:
            n_pos = len(pos_findings)
            candidate_disease_counter = Counter(itertools.chain.from_iterable(
                (self.finding2disease[f] for f in pos_findings)))
            candidate_diseases = [
                disease for disease, count in candidate_disease_counter.items() if count == n_pos]
            # assert len(candidate_diseases) > 0, f'{candidate_disease_counter}'
            candidate_findings = list(set(itertools.chain.from_iterable(
                [self.disease2finding[d] for d in candidate_diseases])))
            self.visited[pos_bytes] = candidate_findings
            return candidate_findings

    def _update_obs(self):
        candidate_findings = self.get_candidate_finding_index()

        mask = np.zeros(self.action_size)
        # Potentially legal actions
        mask[candidate_findings] = 1
        # Illegal actions, including all visited action, no matter pos or neg
        mask[np.argwhere(self.obs != 0).ravel()] = 0
        # Diagnosis actions
        mask[self.n_all_findings:] = 1

        self.obs_dict = {
            "action_mask": mask,
            "obs": self.obs,
            # "avail_actions": np.ones(self.action_size),
            # "avail_actions": mask
        }

    def reset(self, i=None):
        self.obs = np.zeros(self.n_all_findings)
        self.current_step = 0
        if i is None:
            self.disease, self.findings, first_finding = self.one_disease_sample()
            self.obs[first_finding] = 1.0
        else:
            case = self.test_data[i]
            self.disease = case['disease_tag']
            self.findings = [
                f for f, b in case['goal']['implicit_inform_slots'].items() if b]
            for f, b in case['goal']['explicit_inform_slots'].items():
                self.obs[f] = 1.0 if b else -1.0

        if self.mask_actions:
            self._update_obs()
        return self.obs_dict

    def step(self, action):
        # Inquire action
        self.current_step += 1
        if action < self.n_all_findings:
            done = False
            if self.aux_reward:
                ones = (self.obs == 1.0).sum()
                if action in self.findings:
                    if self.obs[action] == 1.0:
                        reward = (self.gamma * ones - ones) * self.lambda_
                    else:
                        reward = (self.gamma * (ones + 1) - ones) * \
                            self.lambda_
                else:
                    reward = (self.gamma * ones - ones) * self.lambda_

            else:
                reward = 0
            self.obs[action] = 1.0 if action in self.findings else -1.0
            # if (self.obs != 0.0).sum() == self.args.max_episode_len + 1:
            #     correctness = self.inference()
            #     reward += 1 if correctness[0] else -1
            #     done = True
            if self.current_step >= self.max_episode_len:
                reward = -1
                done = True

        # Prediction action for single diagnosis action
        elif self.single_diag_action:
            done = True
            # reward = 0
            # if self.aux_reward and self.diag_aux:
            #     ones = (self.obs == 1.0).sum().item()
            #     reward += (self.gamma * ones - ones) * self.lambda_

            correctness = self.inference()
            reward = 1 if correctness[0] else -1

        # Prediction action for direct diagnosis prediction
        else:
            done = True
            prediction = action - self.n_all_findings
            reward = 1 if prediction == self.disease else -1

        if self.mask_actions:
            self._update_obs()
        return self.obs_dict, reward, done, {}

    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        self.np_random, seed = seeding.np_random(seed)

    def inference(self, pos_findings=None, neg_findings=None):
        disease_probs = self.compute_disease_probs(
            pos_findings=pos_findings, neg_findings=neg_findings)
        top5 = disease_probs.argsort()[-5:][::-1]
        return (self.disease == top5[0], self.disease in top5[:3], self.disease in top5)

    def compute_disease_probs(self, pos_findings=None, neg_findings=None, normalize=False):
        """ Make diagnosis prediction given current observed findings """
        if pos_findings is None:
            pos_findings = np.argwhere(self.obs == 1).ravel()
        if neg_findings is None:
            neg_findings = np.argwhere(self.obs == -1).ravel()

        p_f_d = np.empty(self.n_all_diseases)
        for disease in range(self.n_all_diseases):
            prob = 1.0
            flag = False
            f4d = self.disease2finding[disease]

            # Compute negative findings first then the positives
            for finding in neg_findings:
                prob *= 1 - f4d.get(finding, 0)

            for finding in pos_findings:
                if finding in f4d:
                    prob *= f4d[finding]
                else:
                    flag = True
                    break
            if flag:
                prob = 0.0

            p_f_d[disease] = prob
        disease_probs = p_f_d * self.p_d

        if normalize:
            joint_prob = sum(disease_probs)
            if joint_prob == 0.0:
                return 0, 0
            else:
                disease_probs /= joint_prob
                return disease_probs, joint_prob
        else:
            return disease_probs


class QMRWrapper():
    def __init__(self, config):
        self.env = QMR(config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.running_reward = 0

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs_dict, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return obs_dict, score, done, info

    def get_state(self):
        return deepcopy(self.env), self.running_reward

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        return self.env.obs_dict
