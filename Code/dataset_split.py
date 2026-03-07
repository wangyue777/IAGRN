import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample

from utils import Network_Statistic

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, default=0.67, help='the ratio of the training set')
parser.add_argument('--num', type=int, default=500, help='network scale')
parser.add_argument('--p_val', type=float, default=0.5, help='the position of the target with degree equaling to one')
parser.add_argument('--data', type=str, default='hESC', help='data type')
parser.add_argument('--net', type=str, default='Specific', help='network type')
parser.add_argument('--seed', type=int, default=None, help='random seed for reproducible splits')
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

SPECIAL_TRAIN_RATIO = 0.60
SPECIAL_VAL_RATIO = 0.20
SPECIAL_TEST_RATIO = 0.20


def _print_phase_summary(phase: str, pos_count: int, neg_count: int) -> None:
    total = pos_count + neg_count
    print(f"[{phase}] positives: {pos_count}, negatives: {neg_count}, total: {total}")


def _count_labels(df):
    if df is None or df.empty:
        return 0, 0
    pos = int((df['Label'] == 1).sum())
    neg = int((df['Label'] == 0).sum())
    return pos, neg


def _oversample_minority(df, random_state):
    if df.empty or df['Label'].nunique() < 2:
        return df
    pos_df = df[df['Label'] == 1]
    neg_df = df[df['Label'] == 0]
    if len(pos_df) == 0 or len(neg_df) == 0:
        return df
    if len(pos_df) >= len(neg_df):
        return df
    pos_up = resample(pos_df, replace=True, n_samples=len(neg_df), random_state=random_state)
    balanced = pd.concat([neg_df, pos_up], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _prepare_rng(random_state):
    return np.random.default_rng(random_state)


def _shuffle_group(df, rng):
    if df.empty:
        return df
    seed = int(rng.integers(0, 2 ** 32 - 1))
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _split_positive_edges(label_df, random_state):
    if 'TF' not in label_df.columns or 'Target' not in label_df.columns:
        raise ValueError("Label file must contain 'TF' and 'Target' columns.")
    if 'Label' in label_df.columns:
        pos_df = label_df[label_df['Label'] == 1].copy()
    else:
        pos_df = label_df.copy()
    if pos_df.empty:
        return (pd.DataFrame(columns=['TF', 'Target']),
                pd.DataFrame(columns=['TF', 'Target']),
                pd.DataFrame(columns=['TF', 'Target']))

    pos_df[['TF', 'Target']] = pos_df[['TF', 'Target']].astype(np.int64)

    rng = _prepare_rng(random_state)
    train_parts = []
    val_parts = []
    test_parts = []

    for tf_value, group in pos_df.groupby('TF', sort=False):
        group = _shuffle_group(group[['TF', 'Target']], rng)
        total = len(group)
        if total == 0:
            continue

        n_test = int(round(total * SPECIAL_TEST_RATIO))
        n_val = int(round(total * SPECIAL_VAL_RATIO))
        n_test = min(n_test, total)
        n_val = min(n_val, total - n_test)
        n_train = total - n_val - n_test

        if n_train <= 0:
            n_train = 1
            if n_val > n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1

        while n_train + n_val + n_test > total:
            if n_val > n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                n_train -= 1

        while n_train + n_val + n_test < total:
            n_train += 1

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=['TF', 'Target'])
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=['TF', 'Target'])
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=['TF', 'Target'])

    return train_df, val_df, test_df


def _sample_negative_pairs(num_samples, tf_candidates, gene_candidates, forbidden_pairs, used_pairs, rng):
    if num_samples <= 0:
        return []
    neg_pairs = set()
    max_attempts = max(num_samples * 100, 5000)
    attempts = 0
    tf_candidates = np.asarray(tf_candidates, dtype=np.int64)
    gene_candidates = np.asarray(gene_candidates, dtype=np.int64)

    while len(neg_pairs) < num_samples and attempts < max_attempts:
        tf_val = int(rng.choice(tf_candidates))
        target_val = int(rng.choice(gene_candidates))
        attempts += 1
        if tf_val == target_val:
            continue
        pair = (tf_val, target_val)
        if pair in forbidden_pairs or pair in used_pairs or pair in neg_pairs:
            continue
        neg_pairs.add(pair)

    if len(neg_pairs) < num_samples:
        raise RuntimeError("Unable to sample enough unique negative pairs. Try reducing requested negatives or check candidate sets.")

    return list(neg_pairs)


def _build_specific_negative_pools(label_df):
    """Specific"""
    pos_by_tf = {}
    for tf_value, group in label_df.groupby('TF'):
        pos_by_tf[int(tf_value)] = set(int(t) for t in group['Target'].tolist())

    target_frequency = label_df['Target'].value_counts()
    ranked_targets = [int(t) for t in target_frequency.index.tolist()]

    candidate_map = {}
    for tf_value, pos_targets in pos_by_tf.items():
        candidate_map[tf_value] = [t for t in ranked_targets if t not in pos_targets and t != tf_value]

    return pos_by_tf, candidate_map, ranked_targets


def _sample_negative_pairs_specific(num_samples, tf_candidates, gene_candidates, forbidden_pairs,
                                    used_pairs, rng, pos_by_tf, candidate_map, fallback_targets):
    if num_samples <= 0:
        return []

    neg_pairs = set()
    tf_candidates = np.asarray(tf_candidates, dtype=np.int64)
    fallback_targets = np.asarray(fallback_targets, dtype=np.int64)
    gene_candidates = np.asarray(gene_candidates, dtype=np.int64)

    max_attempts = max(num_samples * 200, 10000)
    attempts = 0

    while len(neg_pairs) < num_samples and attempts < max_attempts:
        tf_val = int(rng.choice(tf_candidates))
        preferred = candidate_map.get(tf_val, [])
        if preferred:
            target_pool = np.asarray(preferred, dtype=np.int64)
        else:
            target_pool = fallback_targets if len(fallback_targets) > 0 else gene_candidates

        target_val = int(rng.choice(target_pool))
        attempts += 1

        if target_val == tf_val:
            continue
        pair = (tf_val, target_val)
        if pair in forbidden_pairs or pair in used_pairs or pair in neg_pairs:
            continue

        neg_pairs.add(pair)

    return list(neg_pairs)


def _build_split_dataframe(pos_df, neg_pairs):
    pos_out = pos_df.copy()
    if not pos_out.empty:
        pos_out['Label'] = 1
    else:
        pos_out = pd.DataFrame(columns=['TF', 'Target', 'Label'])

    if neg_pairs:
        neg_df = pd.DataFrame(neg_pairs, columns=['TF', 'Target'])
        neg_df['Label'] = 0
    else:
        neg_df = pd.DataFrame(columns=['TF', 'Target', 'Label'])

    combined = pd.concat([pos_out, neg_df], ignore_index=True)
    return pos_out, neg_df, combined


def _shuffle_df(df, random_state):
    if df.empty:
        return df
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _split_and_sample(label_df, gene_set, tf_set, train_path, val_path,
                      test_path, data_type, random_state, net_type):
    print(f"[SplitPipeline] {data_type}: stratified positives, independent negatives, train oversampling if needed.")

    train_pos_df, val_pos_df, test_pos_df = _split_positive_edges(label_df, random_state)

    rng = _prepare_rng(random_state)
    positive_frames = [df for df in [train_pos_df, val_pos_df, test_pos_df] if not df.empty]
    if positive_frames:
        all_positive_pairs = set(map(tuple, pd.concat(positive_frames, ignore_index=True).values.tolist()))
    else:
        all_positive_pairs = set()

    used_negatives = set()

    if net_type == 'Specific':
        pos_by_tf, candidate_map, ranked_targets = _build_specific_negative_pools(label_df)
        sample_fn = lambda n, used: _sample_negative_pairs_specific(
            n, tf_set, gene_set, all_positive_pairs, used, rng,
            pos_by_tf, candidate_map, ranked_targets
        )
    else:
        sample_fn = lambda n, used: _sample_negative_pairs(n, tf_set, gene_set, all_positive_pairs, used, rng)

    train_neg_pairs = sample_fn(len(train_pos_df), used_negatives)
    used_negatives.update(train_neg_pairs)

    val_neg_pairs = sample_fn(len(val_pos_df), used_negatives)
    used_negatives.update(val_neg_pairs)

    test_neg_pairs = sample_fn(len(test_pos_df), used_negatives)
    used_negatives.update(test_neg_pairs)

    train_pos_out, _, train_df = _build_split_dataframe(train_pos_df, train_neg_pairs)
    val_pos_out, _, val_df = _build_split_dataframe(val_pos_df, val_neg_pairs)
    test_pos_out, _, test_df = _build_split_dataframe(test_pos_df, test_neg_pairs)

    train_df = _oversample_minority(train_df, random_state)

    train_df = _shuffle_df(train_df, random_state)
    val_df = _shuffle_df(val_df, random_state)
    test_df = _shuffle_df(test_df, random_state)

    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    test_df.to_csv(test_path)

    train_pos_path = os.path.join(os.path.dirname(train_path), 'Train_Positive_Edges.csv')
    val_pos_path = os.path.join(os.path.dirname(val_path), 'Validation_Positive_Edges.csv')
    test_pos_path = os.path.join(os.path.dirname(test_path), 'Test_Positive_Edges.csv')

    train_pos_out.to_csv(train_pos_path)
    val_pos_out.to_csv(val_pos_path)
    test_pos_out.to_csv(test_pos_path)

    train_pos_unique = len(train_pos_out)
    val_pos_unique = len(val_pos_out)
    test_pos_unique = len(test_pos_out)

    _print_phase_summary("Training", *_count_labels(train_df))
    print(f"[Training] unique positive edges (pre-oversampling): {train_pos_unique}")
    _print_phase_summary("Validation", *_count_labels(val_df))
    print(f"[Validation] unique positive edges: {val_pos_unique}")
    _print_phase_summary("Test", *_count_labels(test_df))
    print(f"[Test] unique positive edges: {test_pos_unique}")

    return train_pos_path, val_pos_path, test_pos_path


def train_val_test_set(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, density,
                       net_type, data_type, gene_num, p_val=args.p_val):
    print("=================Start dividing the dataset=================")
    print(f"{net_type}\t{data_type}\t{gene_num}")

    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values

    label = pd.read_csv(label_file, index_col=0)
    _split_and_sample(label, gene_set, tf_set, train_set_file, val_set_file, test_set_file,
                      data_type, args.seed, net_type)
    return


def Hard_Negative_Specific_train_test_val(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file,
                                          net_type, data_type, gene_num, ratio=args.ratio, p_val=args.p_val):
    print("=================Start dividing the dataset=================")
    print(f"{net_type}\t{data_type}\t{gene_num}")
    label = pd.read_csv(label_file, index_col=0)
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values

    tf = label['TF'].values
    tf_list = np.unique(tf)

    pos_dict = {i: [] for i in tf_list}
    for i, j in label.values:
        pos_dict[i].append(j)

    neg_dict = {i: [] for i in tf_set}

    for i in tf_set:
        if i in pos_dict.keys():
            pos_item = pos_dict[i]
            pos_item.append(i)
            neg_item = np.setdiff1d(gene_set, pos_item)
            neg_dict[i].extend(neg_item)
            pos_dict[i] = np.setdiff1d(pos_dict[i], i)

        else:
            neg_item = np.setdiff1d(gene_set, i)
            neg_dict[i].extend(neg_item)

    train_pos = {}
    val_pos = {}
    test_pos = {}
    for k in pos_dict.keys():
        if len(pos_dict[k]) == 1:
            p = np.random.uniform(0, 1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]

        elif len(pos_dict[k]) == 2:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:int(len(pos_dict[k]) * ratio)]
            val_pos[k] = pos_dict[k][int(len(pos_dict[k]) * ratio):int(len(pos_dict[k]) * (ratio + 0.1))]
            test_pos[k] = pos_dict[k][int(len(pos_dict[k]) * (ratio + 0.1)):]

    print("----Constructing training set----")
    train_neg = {}
    val_neg = {}
    test_neg = {}
    for k in pos_dict.keys():
        neg_num = len(neg_dict[k])
        np.random.shuffle(neg_dict[k])
        train_neg[k] = neg_dict[k][:int(neg_num * ratio)]
        val_neg[k] = neg_dict[k][int(neg_num * ratio):int(neg_num * (0.1 + ratio))]
        test_neg[k] = neg_dict[k][int(neg_num * (0.1 + ratio)):]

    train_pos_set = []
    for k in train_pos.keys():
        for val in train_pos[k]:
            train_pos_set.append([k, val])

    train_neg_set = []
    for k in train_neg.keys():
        for val in train_neg[k]:
            train_neg_set.append([k, val])

    train_set = train_pos_set + train_neg_set
    train_label = [1 for _ in range(len(train_pos_set))] + [0 for _ in range(len(train_neg_set))]

    train_sample = np.array(train_set)
    train = pd.DataFrame()
    train['TF'] = train_sample[:, 0]
    train['Target'] = train_sample[:, 1]
    train['Label'] = train_label
    train.to_csv(train_set_file)
    train_pos_df = pd.DataFrame(train_pos_set, columns=['TF', 'Target']) if train_pos_set else pd.DataFrame(columns=['TF', 'Target'])
    print('=================Training set partitioning completed=================')
    _print_phase_summary("Training", len(train_pos_set), len(train_neg_set))

    print("----Constructing validation set----")
    val_pos_set = []
    for k in val_pos.keys():
        for val in val_pos[k]:
            val_pos_set.append([k, val])

    val_neg_set = []
    for k in val_neg.keys():
        for val in val_neg[k]:
            val_neg_set.append([k, val])

    val_set = val_pos_set + val_neg_set
    val_label = [1 for _ in range(len(val_pos_set))] + [0 for _ in range(len(val_neg_set))]

    val_sample = np.array(val_set)
    val = pd.DataFrame()
    val['TF'] = val_sample[:, 0]
    val['Target'] = val_sample[:, 1]
    val['Label'] = val_label
    val.to_csv(val_set_file)
    val_pos_df = pd.DataFrame(val_pos_set, columns=['TF', 'Target']) if val_pos_set else pd.DataFrame(columns=['TF', 'Target'])
    print('=================Validation set partitioning completed=================')
    _print_phase_summary("Validation", len(val_pos_set), len(val_neg_set))

    print("----Constructing test set----")
    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])

    test_neg_set = []
    for k in test_neg.keys():
        for j in test_neg[k]:
            test_neg_set.append([k, j])

    test_set = test_pos_set + test_neg_set
    test_label = [1 for _ in range(len(test_pos_set))] + [0 for _ in range(len(test_neg_set))]

    test_sample = np.array(test_set)
    test = pd.DataFrame()
    test['TF'] = test_sample[:, 0]
    test['Target'] = test_sample[:, 1]
    test['Label'] = test_label
    test.to_csv(test_set_file)
    test_pos_df = pd.DataFrame(test_pos_set, columns=['TF', 'Target']) if test_pos_set else pd.DataFrame(columns=['TF', 'Target'])
    print('=================Test set partitioning completed=================')
    _print_phase_summary("Test", len(test_pos_set), len(test_neg_set))

    train_pos_path = os.path.join(os.path.dirname(train_set_file), 'Train_Positive_Edges.csv')
    val_pos_path = os.path.join(os.path.dirname(val_set_file), 'Validation_Positive_Edges.csv')
    test_pos_path = os.path.join(os.path.dirname(test_set_file), 'Test_Positive_Edges.csv')

    train_pos_df.to_csv(train_pos_path)
    val_pos_df.to_csv(val_pos_path)
    test_pos_df.to_csv(test_pos_path)


if __name__ == '__main__':
    net_type = args.net
    data_type = args.data
    gene_num = args.num

    density = Network_Statistic(data_type=data_type, net_scale=gene_num, net_type=net_type)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    benchmark_dir = os.path.join(base_dir, 'Dataset', 'Benchmark Dataset', f'{net_type} Dataset', data_type,
                                 f'TFs+{gene_num}')

    TF2file = os.path.join(benchmark_dir, 'TF.csv')
    Gene2file = os.path.join(benchmark_dir, 'Target.csv')
    label_file = os.path.join(benchmark_dir, 'Label.csv')

    train_dir = os.path.join(base_dir, 'Dataset', 'train', net_type, f'{data_type} {gene_num}')
    val_dir = os.path.join(base_dir, 'Dataset', 'val', net_type, f'{data_type} {gene_num}')
    test_dir = os.path.join(base_dir, 'Dataset', 'test', net_type, f'{data_type} {gene_num}')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_set_file = os.path.join(train_dir, 'Train_set.csv')
    val_set_file = os.path.join(val_dir, 'Validation_set.csv')
    test_set_file = os.path.join(test_dir, 'Test_set.csv')

    if net_type == 'Specific':
        Hard_Negative_Specific_train_test_val(label_file, Gene2file, TF2file, train_set_file, val_set_file,
                                              test_set_file, net_type, data_type, gene_num)
    else:
        train_val_test_set(label_file, Gene2file, TF2file, train_set_file, val_set_file, test_set_file, density,
                           net_type, data_type, gene_num)

