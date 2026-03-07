import argparse
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from interleaved_attention import IAGRN
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation, compute_laplacian_positional_encoding

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, '..'))
RESULT_DIR = os.path.join(PROJECT_ROOT, 'Result')
RESULT_FILE = os.path.join(CODE_DIR, 'results.txt')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=4e-4, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=97, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3, 3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=list, default=[128, 64, 32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=84, help='Random seed')
parser.add_argument('--Type', type=str, default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--esa_layers', type=int, default=3, help='Number of alternating MAB/SAB layers for IAGRN.')
parser.add_argument('--esa_dropout', type=float, default=0.1, help='Dropout rate inside ESA encoder.')
parser.add_argument('--esa_ffn_ratio', type=float, default=2.0, help='Feed-forward expansion ratio inside ESA blocks.')
parser.add_argument('--esa_residual_scale', type=float, default=1.0, help='Residual scaling factor inside ESA blocks.')
parser.add_argument('--mab_heads', type=int, default=4, help='Override MAB attention head number.')
parser.add_argument('--sab_heads', type=int, default=4, help='Override SAB attention head number.')
parser.add_argument('--esa_disable_self_loop', action='store_true', help='Disable adding implicit self-loops inside ESA masks.')
parser.add_argument('--sab_bias_weight', type=float, default=0.1, help='Distance-based bias weight for SAB layers.')
parser.add_argument('--mask_distance_decay', action='store_true', help='Apply -inf to unreachable nodes in SAB bias.')
parser.add_argument('--sce_weight', type=float, default=0.3, help='Weight for scaled cosine error regularization on positive pairs.')
parser.add_argument('--early_stop_patience', type=int, default=0, help='Early stopping patience (epochs).')
parser.add_argument('--early_stop_min_delta', type=float, default=5e-4)
parser.add_argument('--disable_lap_pe', dest='enable_lap_pe', action='store_false')
parser.set_defaults(enable_lap_pe=True)
parser.add_argument('--lap_pe_dim', type=int, default=16)
parser.add_argument('--lap_pe_normalized', action='store_true', default=True)
parser.add_argument('--lap_pe_unormed', dest='lap_pe_normalized', action='store_false')
parser.add_argument('--gene_num', type=int, default=500)
parser.add_argument('--net_types', type=str, nargs='+', default=['Non-Specific'])
args = parser.parse_args()

seed = args.seed
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


def _make_worker_init_fn(base_seed: int):
    def _seed_worker(worker_id: int):
        worker_seed = (base_seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _seed_worker


def write_to_txt(net_type, cell_type, gene_num, auroc, auprc, filename=RESULT_FILE):
    with open(filename, mode='a') as file:
        if file.tell() == 0:
            file.write("Experimental results record\n")
            file.write("=======================================")
        file.write(f"\n{net_type}\t{cell_type}\t{gene_num}\t{auroc:.4f}\t{auprc:.4f}")

def embed2file(tf_embed, tg_embed, gene_file, tf_path, target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed, index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)

def _build_paths(net_type, cell_type, gene_num):
    benchmark_dir = os.path.join(
        PROJECT_ROOT,
        'Dataset',
        'Benchmark Dataset',
        f'{net_type} Dataset',
        cell_type,
        f'TFs+{gene_num}'
    )
    split_root = os.path.join(PROJECT_ROOT, 'Dataset')
    train_file = os.path.join(split_root, 'train', net_type, f'{cell_type} {gene_num}', 'Train_set.csv')
    train_pos_file = os.path.join(split_root, 'train', net_type, f'{cell_type} {gene_num}', 'Train_Positive_Edges.csv')
    val_file = os.path.join(split_root, 'val', net_type, f'{cell_type} {gene_num}', 'Validation_set.csv')
    test_file = os.path.join(split_root, 'test', net_type, f'{cell_type} {gene_num}', 'Test_set.csv')

    exp_file = os.path.join(benchmark_dir, 'BL--ExpressionData.csv')
    tf_file = os.path.join(benchmark_dir, 'TF.csv')
    target_file = os.path.join(benchmark_dir, 'Target.csv')

    for required in [exp_file, tf_file, target_file, train_file, train_pos_file, val_file, test_file]:
        if not os.path.exists(required):
            raise FileNotFoundError(f"Required file not found: {required}")

    return exp_file, tf_file, target_file, train_file, val_file, test_file, train_pos_file

def prepare_dataset(net_type, cell_type, gene_num):
    split_script = os.path.join(CODE_DIR, 'dataset_split.py')
    cmd = [
        sys.executable,
        split_script,
        '--net', net_type,
        '--data', cell_type,
        '--num', str(gene_num)
    ]
    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed)])
    print("=================Start dataset split=================")
    print(f"{net_type}\t{cell_type}\t{gene_num}")
    subprocess.run(cmd, check=True)
    print("=================Dataset split finished=================")

def Running_Model(net_type, cell_type, gene_num, record_embeddings):
    prepare_dataset(net_type, cell_type, gene_num)
    exp_file, tf_file, target_file, train_file, val_file, test_file, train_pos_file = _build_paths(net_type, cell_type, gene_num)

    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()

    tf_indices = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_data = pd.read_csv(train_file, index_col=0).values
    validation_np = pd.read_csv(val_file, index_col=0).values
    test_np = pd.read_csv(test_file, index_col=0).values

    train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)

    train_pos_df = pd.read_csv(train_pos_file, index_col=0)
    if not {'TF', 'Target'}.issubset(train_pos_df.columns):
        raise ValueError("Train_Positive_Edges.csv must contain 'TF' and 'Target' columns.")
    train_pos_np = train_pos_df[['TF', 'Target']].to_numpy(dtype=np.int64)
    pos_labels = np.ones((train_pos_np.shape[0], 1), dtype=np.float32)
    train_pos_edges = np.concatenate([train_pos_np, pos_labels], axis=1)

    graph_dataset = scRNADataset(train_pos_edges, feature.shape[0], flag=False)
    adj_sp = graph_dataset.Adj_Generate(torch.from_numpy(tf_indices), loop=args.loop)

    if args.enable_lap_pe and args.lap_pe_dim > 0:
        lap_pe = compute_laplacian_positional_encoding(
            adj_sp,
            num_vectors=args.lap_pe_dim,
            normalized=args.lap_pe_normalized
        )
        if lap_pe.shape[1] != args.lap_pe_dim:
            pad_width = max(0, args.lap_pe_dim - lap_pe.shape[1])
            if pad_width > 0:
                lap_pe = np.pad(lap_pe, ((0, 0), (0, pad_width)), mode='constant')
        feature = np.concatenate([feature, lap_pe], axis=1)

    data_feature = torch.from_numpy(feature).to(device)

    adj = adj2saprse_tensor(adj_sp).to(device)

    val_data = torch.from_numpy(validation_np).to(device)
    test_data = torch.from_numpy(test_np).to(device)

    model = IAGRN(
        input_dim=feature.shape[1],
        hidden1_dim=args.hidden_dim[0],
        hidden2_dim=args.hidden_dim[1],
        hidden3_dim=args.hidden_dim[2],
        output_dim=args.output_dim,
        num_head1=args.num_head[0],
        num_head2=args.num_head[1],
        alpha=args.alpha,
        device=device,
        type=args.Type,
        num_layers=args.esa_layers,
        dropout=args.esa_dropout,
        ffn_ratio=args.esa_ffn_ratio,
        add_self_loop=not args.esa_disable_self_loop,
        mab_heads_override=args.mab_heads,
        sab_heads_override=args.sab_heads,
        residual_scale=args.esa_residual_scale,
        sab_bias_weight=args.sab_bias_weight,
        mask_distance_decay=args.mask_distance_decay
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    model_path = os.path.join(CODE_DIR, 'model', net_type, f'{cell_type} {gene_num}')
    os.makedirs(model_path, exist_ok=True)
    best_model_path = os.path.join(model_path, 'best_model.pkl')
    log_file = os.path.join(model_path, 'training_log.txt')
    with open(log_file, mode='w') as f:
        f.write("epoch\ttrain_loss\tval_AUC\tval_AUPR\n")

    best_val_auc = -float('inf')
    best_metrics = (0.0, 0.0)
    patience = max(args.early_stop_patience, 0)
    min_delta = max(args.early_stop_min_delta, 0.0)
    epochs_no_improve = 0
    early_stop_epoch = None

    print(f"The cells selected were: {cell_type}, The regulatory network selected is: {net_type}, The number of genes selected is: {gene_num}")
    print("============================Start training============================")

    base_loader_seed = (args.seed if args.seed is not None else torch.initial_seed()) % (2 ** 32)
    data_generator = torch.Generator()

    for epoch in range(args.epochs):
        running_loss = 0.0
        epoch_seed = (base_loader_seed + epoch) % (2 ** 32)
        data_generator.manual_seed(epoch_seed)
        worker_init_fn = _make_worker_init_fn(epoch_seed)
        data_loader = DataLoader(
            train_load,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=data_generator
        )
        for train_x, train_y in data_loader:
            model.train()
            optimizer.zero_grad()

            train_x = train_x.to(device)
            label_vector = train_y.to(device)
            if args.flag:
                train_y = label_vector
            else:
                train_y = label_vector.view(-1, 1)

            pred = model(data_feature, adj, train_x)
            pred = torch.softmax(pred, dim=1) if args.flag else torch.sigmoid(pred)
            loss_BCE = F.binary_cross_entropy(pred, train_y)

            if args.sce_weight > 0:
                tf_embed_batch = model.tf_ouput[train_x[:, 0]]
                target_embed_batch = model.target_output[train_x[:, 1]]
                cosine_sim = F.cosine_similarity(tf_embed_batch, target_embed_batch, dim=1)
                if args.flag:
                    pos_mask = label_vector[:, -1] > 0.5
                else:
                    pos_mask = label_vector.view(-1) > 0.5
                if torch.any(pos_mask):
                    sce_loss = 1 - cosine_sim[pos_mask].mean()
                else:
                    sce_loss = torch.zeros(1, device=device, dtype=loss_BCE.dtype)
                loss = loss_BCE + args.sce_weight * sce_loss
            else:
                loss = loss_BCE

            loss.backward()
            optimizer.step()

            running_loss += loss_BCE.item()

        scheduler.step()

        with torch.no_grad():
            model.eval()
            score = model(data_feature, adj, val_data)
            score = torch.softmax(score, dim=1) if args.flag else torch.sigmoid(score)
            AUC, AUPR, _ = Evaluation(y_pred=score, y_true=val_data[:, -1], flag=args.flag)
            print(f"Epoch:{epoch + 1}\ttrain loss:{running_loss:.4f}\tAUC:{AUC:.3f}\tAUPR:{AUPR:.3f}")
            with open(log_file, mode='a') as f:
                f.write(f"{epoch + 1}\t{running_loss:.4f}\t{AUC:.4f}\t{AUPR:.4f}\n")
            if (AUC - best_val_auc) > min_delta:
                best_val_auc = AUC
                best_metrics = (AUC, AUPR)
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                if patience > 0:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        early_stop_epoch = epoch + 1
                        print(f"[EarlyStopping] No improvement for {patience} epochs, stop at epoch {early_stop_epoch}.")
                        break

    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    if early_stop_epoch is not None:
        print(f"Training terminated early at epoch {early_stop_epoch} based on validation AUC.")

    print("============================Validation============================\n")
    print(f"the best AUROC is: {best_metrics[0]:.3f}, the best AUPRC is: {best_metrics[1]:.3f}")

    print("\n============================Test============================\n")
    model.eval()
    score_test = model(data_feature, adj, test_data)
    score_test = torch.softmax(score_test, dim=1) if args.flag else torch.sigmoid(score_test)
    AUROC, AUPR, _ = Evaluation(y_pred=score_test, y_true=test_data[:, -1], flag=args.flag)
    print(f"the AUROC is: {AUROC:.3f}, the AUPRC is: {AUPR:.3f}")
    write_to_txt(net_type, cell_type, gene_num, AUROC, AUPR)

    if record_embeddings:
        os.makedirs(os.path.join(RESULT_DIR, net_type, f'{cell_type} {gene_num}'), exist_ok=True)
        tf_embed_path = os.path.join(RESULT_DIR, net_type, f'{cell_type} {gene_num}', 'Channel1.csv')
        target_embed_path = os.path.join(RESULT_DIR, net_type, f'{cell_type} {gene_num}', 'Channel2.csv')
        tf_embed, target_embed = model.get_embedding()
        embed2file(tf_embed, target_embed, target_file, tf_embed_path, target_embed_path)

if __name__ == '__main__':
    net_types = args.net_types
    cell_types = ["hESC", "hHEP", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"]
    # cell_types = ["mESC"]
    gene_num = args.gene_num

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(RESULT_FILE, mode='a') as file:
        file.write("\n=======================================")

    for net_type in net_types:
        for idx, cell_type in enumerate(cell_types, start=1):
            record_embeddings = idx == len(cell_types)
            Running_Model(net_type, cell_type, gene_num, record_embeddings)
