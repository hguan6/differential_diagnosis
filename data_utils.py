from collections import defaultdict, Counter
import logging
import numpy as np
import pickle
import json
import os

logger = logging.getLogger(__name__)


def _HPO_postprocess(data):
    for disease, findings in data.items():
        for finding in findings:
            if isinstance(finding[1], list):
                finding[1] = finding[1][1]


def load_graph_data(args):
    dataset_name = 'SymCAT' if args.dataset_name.lower() == 'symcat' else 'HPO'
    n_diseases = args.n_diseases
    dirname = os.path.dirname(__file__)
    with open(f'{dirname}/{args.input_dir}/{dataset_name}.json') as f:
        data = json.load(f)
    with open(f'{dirname}/{args.input_dir}/disease{n_diseases}_{dataset_name}.txt', 'r') as f:
        lines = f.read().splitlines()
    data = {key: data[key] for key in lines}
    if dataset_name == 'HPO':
        _HPO_postprocess(data)

    # Post processes
    disease2finding = defaultdict(dict)
    finding2disease = defaultdict(set)
    for d_idx, (_, findings) in enumerate(data.items()):
        for finding in findings:
            disease2finding[d_idx][finding[0]] = finding[1]
            finding2disease[finding[0]].add(d_idx)
    all_findings = list(finding2disease)
    assert len(disease2finding) == n_diseases

    for disease, findings in disease2finding.items():
        disease2finding[disease] = {all_findings.index(
            finding): prob for finding, prob in findings.items()}

    finding2disease = {all_findings.index(
        finding): list(diseases) for finding, diseases in finding2disease.items()}

    logger.info(
        f'#diseases: {n_diseases}, #findings: {len(finding2disease)}')

    # Assume that all disease are equally likely to happen
    p_d = np.ones(n_diseases) / n_diseases
    return (finding2disease, disease2finding, p_d, None)


def load_dialogue_data(args):
    def get_test_data(data, all_findings, all_diseases):
        for case in data:
            case['disease_tag'] = all_diseases.index(case['disease_tag'])
            case['goal']['implicit_inform_slots'] = {
                all_findings.index(f): b for f, b in case['goal']['implicit_inform_slots'].items()
            }
            case['goal']['explicit_inform_slots'] = {
                all_findings.index(f): b for f, b in case['goal']['explicit_inform_slots'].items()
            }
        return data

    def build_graph(data):
        disease2finding = defaultdict(list)
        finding2disease = defaultdict(set)

        for case in data:
            disease = case['disease_tag']
            for symp, b in case['goal']['implicit_inform_slots'].items():
                if b:
                    disease2finding[disease].append(symp)
                    finding2disease[symp].add(disease)
            for symp, b in case['goal']['explicit_inform_slots'].items():
                if b:
                    disease2finding[disease].append(symp)
                    finding2disease[symp].add(disease)

        for d, fs in disease2finding.items():
            f_len = len(fs)
            disease2finding[d] = {f: round(count / f_len, 4)
                                  for f, count in Counter(fs).items()}

        return finding2disease, disease2finding

    logger.info(f'{args.dataset_name} dataset')
    dirname = os.path.dirname(__file__)
    if args.dataset_name.lower() == 'dxy':
        path = f'{dirname}/{args.input_dir}/dxy_dataset/dxy_dialog_data_dialog_v2.pickle'
    else:
        path = f'{dirname}/{args.input_dir}/acl2018-mds/acl2018-mds.p'

    data = pickle.load(open(path, 'rb'))
    finding2disease, disease2finding = build_graph(data['train'])
    all_findings, all_diseases = list(finding2disease), list(disease2finding)
    logger.info(
        f'{args.dataset_name}, #diseases:{len(all_diseases)}, #findings:{len(all_findings)}')
    test_data = get_test_data(data['test'], all_findings, all_diseases)

    finding2disease, disease2finding = update_graph(
        finding2disease, disease2finding, all_findings, all_diseases)

    # Compute disease priors
    n_cases = len(data['train'])
    counter = Counter([case['disease_tag'] for case in data['train']])
    p_d = np.asarray([counter[d] / n_cases for d in all_diseases])

    # Assume that all disease are equally likely to happen
    # n_diseases = len(disease2finding)
    # p_d = np.ones(n_diseases) / n_diseases

    return (finding2disease, disease2finding, p_d, test_data)


def update_graph(finding2disease, disease2finding, all_findings, all_diseases):
    """ Convert the symptom and disease to their indices """
    d = {}
    for disease, findings in disease2finding.items():
        d[all_diseases.index(disease)] = {
            all_findings.index(finding): prob for finding, prob in findings.items()}
    disease2finding = d

    finding2disease = {all_findings.index(finding): [all_diseases.index(d) for d in diseases]
                       for finding, diseases in finding2disease.items()}

    return finding2disease, disease2finding


def load_data(args):
    dataset_name = args.dataset_name.lower()
    if dataset_name == 'symcat' or dataset_name == 'hpo':
        return load_graph_data(args)
    elif dataset_name == 'muzhi' or dataset_name == 'dxy':
        return load_dialogue_data(args)
    else:
        raise ValueError(f'Dataset name {dataset_name} does not exist.')
