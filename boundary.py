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

class TimitBoundaryDataset(Dataset):
    def __init__(self, split, data_dir, bucket_size=3, erase_diagnal=1):
        self.split = split
        self.data_dir = data_dir
        self.setname = 'train' if split == 'train' else 'test'
        self.table = pd.read_csv(os.path.join(data_dir, f'{split}.csv'))
        self.features = pickle.load(open(os.path.join(data_dir, f'{split}.pkl'), 'rb'))
        self.erase_diagnal = erase_diagnal

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
            boundary = (np.array(boundary, dtype=np.float32) / 3).round().astype(np.int32)
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

        nodiagnal = copy.deepcopy(alignments)
        if self.erase_diagnal > 0:
            for k in range(maxlen):
                start = max(0, k - self.erase_diagnal + 1)
                end = min(maxlen, k + self.erase_diagnal)
                nodiagnal[:, start:end, k] = 0.0

        return alignments, nodiagnal, weights, boundaries

    def __getitem__(self, idx):
        df = self.X[idx]
        x_batch = []
        y_batch = []
        for idx, row in df.iterrows():
            x = torch.FloatTensor(self.features[row.featureid])
            y = pd.eval(row.feature_boundary)
            x_batch.append(x)
            y_batch.append(y)
        x_batch_pad = pad_sequence(x_batch, batch_first=True)
        x_spec = process_test_MAM_data(spec=(x_batch_pad,))
        alignments, nodiagnal, weights, y_batch = self.batch_boundaries(y_batch)
        batch = {
            'specs': x_spec,
            'alignments': alignments,
            'nodiagnal': nodiagnal,
            'weights': weights,
            'phoneseqs': y_batch
        }
        return batch

    def __len__(self):
        return len(self.X)


class LibriBoundaryDataset(Dataset):
    def __init__(self, split, feature_dir, aligned_dir, bucket_size=3, erase_diagnal=1):
        self.split = split
        self.feature_dir = feature_dir
        self.aligned_dir = aligned_dir
        self.setname = 'train-clean-360' if split == 'train' else 'test-clean'
        self.table = pd.read_csv(os.path.join(feature_dir, f'{self.setname}.csv')).sort_values(by=['length'], ascending=False)
        self.erase_diagnal = erase_diagnal

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

        nodiagnal = copy.deepcopy(alignments)
        if self.erase_diagnal > 0:
            for k in range(maxlen):
                start = max(0, k - self.erase_diagnal + 1)
                end = min(maxlen, k + self.erase_diagnal)
                nodiagnal[:, start:end, k] = 0.0

        return alignments, nodiagnal, weights

    def __getitem__(self, idx):
        filenames = self.X[idx]
        x_batch = []
        y_batch = []
        for filename in filenames:
            x = torch.FloatTensor(np.load(os.path.join(self.feature_dir, self.setname, f'{filename}.npy')))
            y = pickle.load(open(os.path.join(self.aligned_dir, self.setname, f'{filename}.pkl'), 'rb'))
            y = list(y[np.arange(0, len(y), 3)])
            x_batch.append(x)
            y_batch.append(y)
        x_batch_pad = pad_sequence(x_batch, batch_first=True)
        x_spec = process_test_MAM_data(spec=(x_batch_pad,))
        alignments, nodiagnal, weights = self.batch_seqs(y_batch)
        batch = {
            'specs': x_spec,
            'alignments': alignments,
            'nodiagnal': nodiagnal,
            'weights': weights,
            'phoneseqs': y_batch
        }
        return batch

    def __len__(self):
        return len(self.X)


class Scalars(nn.Module):
    def __init__(self, num_scalar):
        super(Scalars, self).__init__()
        self.scalars = nn.Parameter(torch.zeros(num_scalar))

    def forward(self, attentions, alignments, weights):
        # attentions: (bsx, num_layer, num_head, maxlen, maxlen)
        # alignments: (bsx, maxlen, maxlen)
        # weights: (bsx, maxlen, maxlen)
        bsx = attentions.size(0)
        maxlen = attentions.size(-1)
        attentions = attentions.permute(0, 3, 4, 1, 2)
        # attentions: (bsx, maxlen, maxlen, num_layer, num_head)
        attentions = attentions.reshape(bsx * maxlen * maxlen, 1, -1)
        attn_weights = F.softmax(self.scalars, dim=-1).view(1, -1, 1).expand(bsx * maxlen * maxlen, -1, -1)
        logits = torch.bmm(attentions, attn_weights).squeeze().view(bsx, maxlen, maxlen)
        # logits: (bsx, maxlen, maxlen)
        loss = F.binary_cross_entropy(logits, alignments, weight=alignments)
        return loss, logits


def test(args, model, mockingjay, trainloader, testloader, device='cuda'):
    with torch.no_grad():
        model.eval()
        loss_sum = 0
        num_batch = 0
        for batch in tqdm(testloader):
            attentions, _ = mockingjay.forward(spec=batch['specs'], all_layers=True, tile=False, process_from_loader=True)
            alignments = batch['alignments']
            nodiagnal = batch['nodiagnal'].to(device=device)
            weights = batch['weights'].to(device=device)
            attentions, alignments, nodiagnal, weights = resize([attentions, alignments, nodiagnal, weights])
            loss, logits = model(attentions, alignments, weights)
            loss_sum += loss.detach().cpu().item()
            num_batch += 1
        loss_mean = loss_sum / num_batch
        print(loss_mean)
    return loss_mean


def visual_attnmap(args, model, mockingjay, trainloader, testloader, device='cuda'):
    with torch.no_grad():
        model.eval()
        test_len = testloader.dataset.__len__()
        train_len = trainloader.dataset.__len__()
        test_indices = np.arange(0, test_len, test_len // 5)
        train_indices = np.arange(0, train_len, train_len // 5)
        for idx in test_indices:
            batch = testloader.dataset[idx]
            attentions, _ = mockingjay.forward(spec=batch['specs'], all_layers=True, tile=False, process_from_loader=True)
            alignments = batch['alignments']
            nodiagnal = batch['nodiagnal'].to(device=device)
            weights = batch['weights'].to(device=device)
            attentions, alignments, nodiagnal, weights = resize([attentions, alignments, nodiagnal, weights])
            loss, logits = model(attentions, nodiagnal, weights)
            torch.save(logits.detach().cpu(), os.path.join(args.exppath, f'test{idx}.logit'))
            torch.save(alignments.detach().cpu(), os.path.join(args.exppath, f'test{idx}.align'))
            torch.save(nodiagnal.detach().cpu(), os.path.join(args.exppath, f'test{idx}.nodiag'))
        torch.cuda.empty_cache()
        for idx in train_indices:
            batch = trainloader.dataset[idx]
            attentions, _ = mockingjay.forward(spec=batch['specs'], all_layers=True, tile=False, process_from_loader=True)
            alignments = batch['alignments']
            nodiagnal = batch['nodiagnal'].to(device=device)
            weights = batch['weights'].to(device=device)
            attentions, alignments, nodiagnal, weights = resize([attentions, alignments, nodiagnal, weights])
            loss, logits = model(attentions, nodiagnal, weights)
            torch.save(logits.detach().cpu(), os.path.join(args.exppath, f'train{idx}.logit'))
            torch.save(alignments.detach().cpu(), os.path.join(args.exppath, f'train{idx}.align'))
            torch.save(nodiagnal.detach().cpu(), os.path.join(args.exppath, f'train{idx}.nodiag'))


def resize(tensors):
    minlen = min([tensor.size(-1) for tensor in tensors])
    return [tensor[:, :minlen, :minlen] for tensor in tensors]


def train(args, model, mockingjay, trainloader, testloader, device='cuda', train_steps=100000, eval_steps=100):
    opt = optim.Adam(model.parameters(), lr=args.lr)
    step = 0
    train_finished = False
    pbar = tqdm(total=train_steps)
    loss_sum = 0
    loss_best = 100
    accu_num = 0
    while not train_finished:
        for batch in trainloader:
            model.train()
            opt.zero_grad()
            attentions, _ = mockingjay.forward(spec=batch['specs'], all_layers=True, tile=False, process_from_loader=True)
            alignments = batch['alignments']
            nodiagnal = batch['nodiagnal'].to(device=device)
            weights = batch['weights'].to(device=device)
            attentions, alignments, nodiagnal, weights = resize([attentions, alignments, nodiagnal, weights])

            loss, logits = model(attentions, nodiagnal, weights)
            loss.backward()
            if accu_num < args.accumulate:
                accu_num += 1
            else:
                opt.step()
                accu_num = 0

            loss_sum += loss.detach().cpu().item()
            step += 1
            pbar.update(1)
            if step % eval_steps == 0:
                loss_mean = loss_sum / eval_steps
                args.comet.log_metric('train_loss', loss_mean)

                def visualize(comet, tensor, name):
                    fig = plt.figure(figsize=(30, 30))
                    plt.imshow(tensor)
                    comet.log_figure(name, fig)

                visualize(args.comet, logits[0].detach().cpu(), 'train_logit')
                visualize(args.comet, alignments[0].detach().cpu(), 'train_align')
                visualize(args.comet, nodiagnal[0].detach().cpu(), 'train_nodiag')

                loss_sum = 0
                if loss_mean < loss_best:
                    loss_best = loss_mean
                    torch.save({
                        'model_state_dict': model.state_dict()
                    }, os.path.join(args.exppath, f'{str(loss_best)[:10]}.pth'))
            if step == train_steps:
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
    parser.add_argument('--num_scalar', default=144, type=int)
    parser.add_argument('--libri_feature_dir', default='data/libri_mel160_subword5000', type=str)
    parser.add_argument('--libri_aligned_dir', default='data/libri_phone', type=str)
    parser.add_argument('--timit_dir', default='data/timit_mel160_phoneme63_aligned', type=str)
    parser.add_argument('--bucket_size', default=8, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--accumulate', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--erase_diagnal', default=1, type=int)
    parser.add_argument('--dataset', default='libri', type=str)
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
    setattr(args, 'comet', exp)

    return args


def main():
    args = get_preprocess_args()

    if args.dataset == 'libri':
        trainset = LibriBoundaryDataset('train', args.libri_feature_dir, args.libri_aligned_dir, bucket_size=args.bucket_size, erase_diagnal=args.erase_diagnal)
        testset = LibriBoundaryDataset('test', args.libri_feature_dir, args.libri_aligned_dir, bucket_size=args.bucket_size, erase_diagnal=args.erase_diagnal)
    elif args.dataset == 'timit':
        trainset = TimitBoundaryDataset('train', args.timit_dir, bucket_size=args.bucket_size, erase_diagnal=args.erase_diagnal)
        testset = TimitBoundaryDataset('test', args.timit_dir, bucket_size=args.bucket_size, erase_diagnal=args.erase_diagnal)

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=lambda xs: xs[0])
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=lambda xs: xs[0])

    mockingjay = get_mockingjay_model(from_path=args.mock, output_attention=True)
    model = Scalars(args.num_scalar)
    if args.ckpt != '':
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device=args.device)

    if args.mode == 'train':
        train(args, model, mockingjay, trainloader, testloader, device=args.device)
    elif args.mode == 'test':
        test(args, model, mockingjay, trainloader, testloader, device=args.device)
    elif args.mode == 'visual':
        visual_attnmap(args, model, mockingjay, trainloader, testloader, device=args.device)
    else:
        print('Please specify the mode')


if __name__ == '__main__':
    main()
