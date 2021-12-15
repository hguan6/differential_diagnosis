import json
import random
from collections import Counter
import itertools
import numpy as np

from data_utils import load_data


class QMR:
    def __init__(self, config):
        # Set arguments
        self.args = config['args']
        self.max_episode_len = self.args.max_episode_len

        # Set random seed
        random.seed(2021)
        np.random.seed(2021)

        # Load dataset
        data_info = load_data(self.args)
        self.finding2disease, self.disease2finding, self.p_d, self.test_data = data_info
        self.n_all_findings = len(self.finding2disease)
        self.n_all_diseases = len(self.disease2finding)

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

        first_finding = random.choice(list(findings))
        return disease, findings, first_finding

    def update_candidate_diseases(self, finding):
        self.candidate_diseases = [
            d for d in self.candidate_diseases if finding in self.disease2finding[d]]

    def get_candidate_finding_index(self):
        return set(itertools.chain.from_iterable(
            [self.disease2finding[d] for d in self.candidate_diseases]))

    def reset(self, i=None):
        # For simulation data
        if i is None:
            self.disease, self.findings, first_finding = self.one_disease_sample()
            self.pos_findings = [first_finding]
            self.neg_findings = []
            self.candidate_diseases = self.finding2disease[first_finding]
        # For  `real` data
        else:
            case = self.test_data[i]
            self.disease = case['disease_tag']
            self.findings = set(
                f for f, b in case['goal']['implicit_inform_slots'].items() if b)

            # Get pos_findings and neg_findings
            self.pos_findings = []
            self.neg_findings = []
            for f, b in case['goal']['explicit_inform_slots'].items():
                if b:
                    self.pos_findings.append(f)
                    self.findings.add(f)
                else:
                    self.neg_findings.append(f)
            # Get candidate_diseases
            for i, f in enumerate(self.pos_findings):
                if i == 0:
                    self.candidate_diseases = self.finding2disease[f]
                else:
                    self.update_candidate_diseases(f)

    def step(self, action):
        if action in self.findings:
            self.pos_findings.append(action)
            self.update_candidate_diseases(action)
        else:
            self.neg_findings.append(action)

    def inference(self, pos_findings=None, neg_findings=None):
        disease_probs = self.compute_disease_probs(
            pos_findings=pos_findings, neg_findings=neg_findings)
        top5 = disease_probs.argsort()[-5:][::-1]
        return (self.disease == top5[0], self.disease in top5[:3], self.disease in top5)

    def compute_disease_probs(self, pos_findings=None, neg_findings=None, normalize=False):
        """ Make diagnosis prediction given current observed findings """
        if pos_findings is None:
            pos_findings = self.pos_findings
        if neg_findings is None:
            neg_findings = self.neg_findings

        p_f_d = np.empty(self.n_all_diseases)
        for disease in range(self.n_all_diseases):
            prob = 1.0
            f4d = self.disease2finding[disease]

            # Compute negative findings first then the positives
            for finding in neg_findings:
                prob *= 1 - f4d.get(finding, 0)

            for finding in pos_findings:
                if finding in f4d:
                    prob *= f4d[finding]
                else:
                    prob = 0.0
                    break
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
