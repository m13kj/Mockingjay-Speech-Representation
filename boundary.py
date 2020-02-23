from comet_ml import Experiment

import os
import copy
import argparse
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from runner_mockingjay import get_mockingjay_model
from utility.mam import process_train_MAM_data, process_test_MAM_data
from ipdb import set_trace

HALF_BATCHSIZE_SEQLEN = 400


def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def visualize(comet, tensor, name, step=None):
    fig = plt.figure(figsize=(30, 30))
    plt.imshow(tensor)
    comet.log_figure(name, fig, step=step)


class TimitBoundaryDataset(Dataset):
    def __init__(self, split, data_dir, mockcfg, bucket_size=3, erase_diagnal=1):
        self.split = split
        self.data_dir = data_dir
        self.setname = 'train' if split == 'train' else 'test'
        self.table = pd.read_csv(os.path.join(data_dir, f'{split}.csv'))
        self.features = pickle.load(open(os.path.join(data_dir, f'{split}.pkl'), 'rb'))
        self.erase_diagnal = erase_diagnal
        self.mockcfg = mockcfg
        self.downsample_rate = mockcfg['downsample_rate']

        self.X = []
        batch = []
        for idx, row in self.table.iterrows():
            batch.append(row)
            if len(batch) == bucket_size:
                batch = pd.concat(batch, axis=1).transpose()
                if (bucket_size >= 2) and (batch.feature_seqlen.max() > HALF_BATCHSIZE_SEQLEN):
                    self.X.append(batch.iloc[:bucket_size//2])
                    self.X.append(batch.iloc[bucket_size//2:])
                else:
                    self.X.append(batch)
                batch = []

        if len(batch) > 0:
            self.X.append(batch)

    def batch_boundaries(self, boundaries):
        # boundaries: double list
        boundaries_refined = []
        for boundary in boundaries:
            boundary = (np.array(boundary, dtype=np.float32) / self.downsample_rate).round().astype(np.int32)
            boundary_refined = [boundary[0]]
            preloc = boundary[0]
            for loc in boundary[1:]:
                if loc != preloc:
                    preloc = loc
                    boundary_refined.append(loc)
            boundaries_refined.append(boundary_refined)
        boundaries = boundaries_refined

        bsx = len(boundaries)
        maxlen = max([item[-1] for item in boundaries])
        alignments = torch.zeros(bsx, maxlen, maxlen).float()
        weights = torch.zeros(bsx, maxlen, maxlen).float()

        for i, boundary in enumerate(boundaries):
            seqlen = boundary[-1]
            weights[i, :seqlen, :seqlen] = 1.0
            curr_position = 0
            for endpoint in boundary[1:]:
                alignments[i, curr_position:endpoint, curr_position:endpoint] = 1.0
                curr_position = endpoint

        labels = copy.deepcopy(alignments)
        if self.erase_diagnal > 0:
            for k in range(maxlen):
                start = max(0, k - self.erase_diagnal + 1)
                end = min(maxlen, k + self.erase_diagnal)
                labels[:, start:end, k] = 0.0
        weights = copy.deepcopy(labels)

        return alignments, labels, weights, boundaries

    def __getitem__(self, idx):
        df = self.X[idx]
        x_batch = []
        y_batch = []
        id_batch = []
        for idx, row in df.iterrows():
            x = torch.FloatTensor(self.features[row.featureid])
            y = pd.eval(row.feature_boundary)
            x_batch.append(x)
            y_batch.append(y)
            id_batch.append(row.fileid)
        x_batch_pad = pad_sequence(x_batch, batch_first=True)
        x_spec = process_test_MAM_data(spec=(x_batch_pad,), config=self.mockcfg)
        alignments, labels, weights, y_batch = self.batch_boundaries(y_batch)
        batch = {
            'specs': x_spec,
            'alignments': alignments,
            'labels': labels,
            'weights': weights,
            'phoneseqs': y_batch,
            'fileids': id_batch
        }
        return batch

    def __len__(self):
        return len(self.X)


class LibriBoundaryDataset(Dataset):
    def __init__(self, split, feature_dir, aligned_dir, mockcfg, bucket_size=3, erase_diagnal=1):
        self.split = split
        self.feature_dir = feature_dir
        self.aligned_dir = aligned_dir
        self.setname = 'train-clean-360' if split == 'train' else 'test-clean'
        self.table = pd.read_csv(os.path.join(feature_dir, f'{self.setname}.csv')).sort_values(by=['length'], ascending=False)
        self.erase_diagnal = erase_diagnal
        self.mockcfg = mockcfg
        self.downsample_rate = mockcfg['downsample_rate']

        self.X = []
        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        unaligned_list = pickle.load(open(os.path.join(aligned_dir, 'unaligned.pkl'), 'rb'))
        batch_x, batch_len = [], []
        for x, x_len in zip(X, X_lens):
            if x in unaligned_list:
                continue
            batch_x.append(x.split('/')[-1][:-4])
            batch_len.append(x_len)

            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_SEQLEN):
                    self.X.append(batch_x[:bucket_size//2])
                    self.X.append(batch_x[bucket_size//2:])
                else:
                    self.X.append(batch_x)
                batch_x, batch_len = [], []

        # Gather the last batch
        if len(batch_x) > 0:
            self.X.append(batch_x)

    def batch_seqs(self, phoneseqs):
        bsx = len(phoneseqs)
        maxlen = max([len(item) for item in phoneseqs])
        alignments = torch.zeros(bsx, maxlen, maxlen).float()
        weights = torch.zeros(bsx, maxlen, maxlen).float()

        for i, phoneseq in enumerate(phoneseqs):
            seqlen = len(phoneseq)
            weights[i, :seqlen, :seqlen] = 1.0
            prev_phone = phoneseq[0]
            curr_phonelen = 1
            curr_phonepos = 0
            phoneseq.append(-100)
            for j in range(1, seqlen):
                if phoneseq[j] == prev_phone:
                    curr_phonelen += 1
                else:
                    endpos = curr_phonepos + curr_phonelen
                    alignments[i, curr_phonepos:endpos, curr_phonepos:endpos] = 1.0
                    prev_phone = phoneseq[j]
                    curr_phonelen = 1
                    curr_phonepos = j

        labels = copy.deepcopy(alignments)
        if self.erase_diagnal > 0:
            for k in range(maxlen):
                start = max(0, k - self.erase_diagnal + 1)
                end = min(maxlen, k + self.erase_diagnal)
                labels[:, start:end, k] = 0.0
        weights = copy.deepcopy(labels)

        return alignments, labels, weights

    def __getitem__(self, idx):
        filenames = self.X[idx]
        x_batch = []
        y_batch = []
        for filename in filenames:
            x = torch.FloatTensor(np.load(os.path.join(self.feature_dir, self.setname, f'{filename}.npy')))
            y = pickle.load(open(os.path.join(self.aligned_dir, self.setname, f'{filename}.pkl'), 'rb'))
            y = list(y[np.arange(0, len(y), self.downsample_rate)])
            x_batch.append(x)
            y_batch.append(y)
        x_batch_pad = pad_sequence(x_batch, batch_first=True)
        x_spec = process_test_MAM_data(spec=(x_batch_pad,), config=self.mockcfg)
        alignments, labels, weights = self.batch_seqs(y_batch)
        batch = {
            'specs': x_spec,
            'alignments': alignments,
            'labels': labels,
            'weights': weights,
            'phoneseqs': y_batch,
            'fileids': filenames
        }
        return batch

    def __len__(self):
        return len(self.X)


class Scalars(nn.Module):
    def __init__(self, num_scalar):
        super(Scalars, self).__init__()
        self.scalars = nn.Parameter(torch.zeros(num_scalar))

    def forward(self, attentions, labels, weights):
        assert attentions.size(-1) == labels.size(-1) == weights.size(-1)
        # attentions: (bsx, num_layer, num_head, maxlen, maxlen)
        # labels: (bsx, maxlen, maxlen)
        # weights: (bsx, maxlen, maxlen)
        bsx = attentions.size(0)
        maxlen = attentions.size(-1)
        attentions = attentions.permute(0, 3, 4, 1, 2)
        # attentions: (bsx, maxlen, maxlen, num_layer, num_head)
        attentions = attentions.reshape(bsx * maxlen * maxlen, 1, -1)
        attn_weights = F.softmax(self.scalars, dim=-1).view(1, -1, 1).expand(bsx * maxlen * maxlen, -1, -1)
        logits = torch.bmm(attentions, attn_weights).squeeze().view(bsx, maxlen, maxlen)
        # logits: (bsx, maxlen, maxlen)
        loss = F.binary_cross_entropy(logits, labels, weight=weights)
        return loss, logits


def resize(tensors):
    minlen = min([tensor.size(-1) for tensor in tensors])
    tensors_resized = []
    for tensor in tensors:
        assert tensor.dim() == 3 or tensor.dim() == 5
        assert np.abs(tensor.size(-1) - minlen) < 4, f'minlen: {minlen}, while tensor shape: {tensor.shape}'
        tensor_resized = tensor[:, :minlen, :minlen] if tensor.dim() == 3 else tensor[:, :, :, :minlen, :minlen]
        tensors_resized.append(tensor_resized)
    return tensors_resized


def visual(args, model, mockingjay, trainloader, testloader, device='cuda'):
    with torch.no_grad():
        model.eval()
        for split, dataset in zip(['train', 'test'], [trainloader.dataset, testloader.dataset]):
            batch_num = dataset.__len__()
            indices = [-30, -60, -90]
            for idx, indice in enumerate(indices):
                batch = dataset[indice]
                attentions, _ = mockingjay.forward(spec=batch['specs'], all_layers=True, tile=False, process_from_loader=True)
                alignments = batch['alignments']
                labels = batch['labels'].to(device=device)
                weights = batch['weights'].to(device=device)
                fileids = batch['fileids']
                attentions, alignments, labels, weights = resize([attentions, alignments, labels, weights])
                loss, logits = model(attentions, labels, weights)

                name = f'{split}.{fileids[0]}'
                print(f'Dealing with {name}')

                visualize(args.comet, logits[0].detach().cpu(), f'0.{name}.logit', step=idx)
                visualize(args.comet, alignments[0].detach().cpu(), f'1.{name}.align', step=idx)
                visualize(args.comet, labels[0].detach().cpu(), f'2.{name}.label', step=idx)
                visualize(args.comet, weights[0].detach().cpu(), f'3.{name}.weight', step=idx)

                bsx = attentions.size(0)
                maxlen = attentions.size(-1)
                attnmaps = attentions.detach().cpu().permute(1, 2, 0, 3, 4).reshape(-1, bsx, maxlen, maxlen)
                for attnid, attnmap in enumerate(attnmaps):
                    visualize(args.comet, attnmap[0], f'5.{name}.{attnid}.attn', step=idx)


def test(args, model, mockingjay, trainloader, testloader, device='cuda'):
    with torch.no_grad():
        model.eval()
        loss_sum = 0
        num_batch = 0
        for batch in tqdm(testloader):
            attentions, _ = mockingjay.forward(spec=batch['specs'], all_layers=True, tile=False, process_from_loader=True)
            alignments = batch['alignments']
            labels = batch['labels'].to(device=device)
            weights = batch['weights'].to(device=device)
            attentions, alignments, labels, weights = resize([attentions, alignments, labels, weights])
            loss, logits = model(attentions, labels, weights)
            loss_sum += loss.detach().cpu().item()
            num_batch += 1
        loss_mean = loss_sum / num_batch
        print(loss_mean)
    return loss_mean


def train(args, model, mockingjay, trainloader, testloader, device='cuda'):
    opt = optim.Adam(model.parameters(), lr=args.lr)
    step = 0
    train_finished = False
    pbar = tqdm(total=args.train_steps)
    loss_sum = 0
    loss_best = 100
    accu_num = 0
    while not train_finished:
        for batch in trainloader:
            model.train()
            opt.zero_grad()
            attentions, _ = mockingjay.forward(spec=batch['specs'], all_layers=True, tile=False, process_from_loader=True)
            alignments = batch['alignments']
            labels = batch['labels'].to(device=device)
            weights = batch['weights'].to(device=device)
            fileids = batch['fileids']
            attentions, alignments, labels, weights = resize([attentions, alignments, labels, weights])

            loss, logits = model(attentions, labels, weights)
            loss.backward()
            if accu_num < args.accumulate:
                accu_num += 1
            else:
                opt.step()
                accu_num = 0

            loss_sum += loss.detach().cpu().item()
            step += 1
            pbar.update(1)
            if step % args.eval_steps == 0:
                loss_mean = loss_sum / args.eval_steps
                args.comet.log_metric('train_loss', loss_mean)

                visualize(args.comet, logits[0].detach().cpu(), f'0.{fileids[0]}.logit')
                visualize(args.comet, alignments[0].detach().cpu(), f'1.{fileids[0]}.align')
                visualize(args.comet, labels[0].detach().cpu(), f'2.{fileids[0]}.label')
                visualize(args.comet, weights[0].detach().cpu(), f'3.{fileids[0]}.weight')

                loss_sum = 0
                if loss_mean < loss_best:
                    loss_best = loss_mean
                    torch.save({
                        'model_state_dict': model.state_dict()
                    }, os.path.join(args.exppath, f'{str(loss_best)[:10]}.pth'))
            if step == args.train_steps:
                train_finished = True
                break
    pbar.close()


def get_preprocess_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--expname', type=str, required=True)
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--mock', default='result/result_mockingjay/mockingjay_libri_sd1337_LinearLarge/mockingjay-500000.ckpt', type=str)
    parser.add_argument('--ckpt', default='', type=str)
    parser.add_argument('--libri_feature_dir', default='data/libri_mel160_subword5000', type=str)
    parser.add_argument('--libri_aligned_dir', default='data/libri_phone', type=str)
    parser.add_argument('--timit_dir', default='data/timit_mel160_phoneme63_aligned', type=str)
    parser.add_argument('--train_steps', default=100000, type=int)
    parser.add_argument('--eval_steps', default=100, type=int)
    parser.add_argument('--bucket_size', default=8, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--accumulate', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--erase_diagnal', default=3, type=int)
    parser.add_argument('--dataset', default='libri', type=str)
    parser.add_argument('--train_shuffle', default=True, type=boolean_string)
    parser.add_argument('--test_shuffle', default=False, type=boolean_string)
    args = parser.parse_args()

    setattr(args, 'exppath', os.path.join('boundary', args.expname))
    if not os.path.exists(args.exppath):
        os.makedirs(args.exppath)

    exp = Experiment(project_name='boundary',
                     auto_param_logging=False,
                     auto_metric_logging=False,
                     auto_output_logging=None,
                     disabled=args.dryrun)
    exp.set_name(args.expname)
    exp.log_parameters(vars(args))
    setattr(args, 'comet', exp)

    return args


def main():
    args = get_preprocess_args()

    mockingjay, mock_config, mock_paras = get_mockingjay_model(from_path=args.mock, output_attention=True)
    mockcfg = mock_config['mockingjay']
    num_scalar = mockcfg['num_hidden_layers'] * mockcfg['num_attention_heads']
    model = Scalars(num_scalar)
    if args.ckpt != '':
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device=args.device)

    if args.dataset == 'libri':
        trainset = LibriBoundaryDataset('train', args.libri_feature_dir, args.libri_aligned_dir, mockcfg,
                                        bucket_size=args.bucket_size, erase_diagnal=args.erase_diagnal)
        testset = LibriBoundaryDataset('test', args.libri_feature_dir, args.libri_aligned_dir, mockcfg,
                                        bucket_size=args.bucket_size, erase_diagnal=args.erase_diagnal)
    elif args.dataset == 'timit':
        trainset = TimitBoundaryDataset('train', args.timit_dir, mockcfg,
                                        bucket_size=args.bucket_size, erase_diagnal=args.erase_diagnal)
        testset = TimitBoundaryDataset('test', args.timit_dir, mockcfg,
                                        bucket_size=args.bucket_size, erase_diagnal=args.erase_diagnal)

    trainloader = DataLoader(trainset, batch_size=1, shuffle=args.train_shuffle, num_workers=args.num_workers, collate_fn=lambda xs: xs[0])
    testloader = DataLoader(testset, batch_size=1, shuffle=args.test_shuffle, num_workers=args.num_workers, collate_fn=lambda xs: xs[0])

    handle = globals()[args.mode]
    handle(args, model, mockingjay, trainloader, testloader, device=args.device)


if __name__ == '__main__':
    main()
