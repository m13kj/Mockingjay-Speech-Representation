# -*- coding: utf-8 -*- #

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from utility.asr import encode_target
from utility.audio import extract_feature, mel_dim, num_freq
from ipdb import set_trace


def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def truncate_path(path, mode):
    assert mode == 'train' or mode == 'test', 'mode should be train/test'
    # input path might be an absoulute or relative path, but we only need information about the file location INSIDE the timit dataset. hence, truncate the original path to START WITH 'train' or 'test'

    path = str(path).split('/')
    path = path[path.index(mode):]
    path = '/'.join(path)
    path = '.'.join(path.split('.')[:-1])  # drop file extension
    return path


def read_text(file, target):
    boundaries = [0]
    labels = []
    if target == 'phoneme':
        with open(file.replace('.wav', '.phn'), 'r') as f:
            for line in f:
                linelist = line.replace('\n', '').split(' ')
                boundaries.append(int(linelist[1]))
                labels.append(linelist[2])
    elif target in ['char', 'subword', 'word']:
        with open(file.replace('.wav', '.wrd'), 'r') as f:
            for line in f:
                labels.append(line.replace('\n', '').split(' ')[-1])
        if target == 'char':
            labels = [c for c in ' '.join(labels)]
    else:
        raise ValueError('Unsupported target: ' + target)
    return labels, boundaries


def preprocess(args, dim, phonesets):
    encode_table = phonesets['large']['phone2id']
    output_dir = os.path.join(
        args.output_path, '_'.join([
            'timit',
            str(args.feature_type) + str(dim),
            str(args.target) + str(len(encode_table)),
            'aligned'
        ]))
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    for mode in ['train', 'test']:
        print(f'Preprocessing {mode} data...', end='')
        all_audio = list(
            Path(os.path.join(args.data_path, mode)).rglob("*.[wW][aA][vV]"))
        print(f'{len(all_audio)} audio files found')

        if mode == 'test':
            with open(args.testspk_path, 'r') as f:
                test_spk = [item[:-1] for item in f.readlines()]

        todo = []
        spks = []
        for pth in all_audio:
            pth_str = str(pth).split('/')
            spk = pth_str[-2]
            txt_type = pth_str[-1][:2]
            if txt_type == 'sa':
                continue
            if mode == 'test' and spk not in test_spk:
                continue
            else:
                if spk not in spks:
                    spks.append(spk)
                todo.append(pth)
        print(f'{len(spks)} speakers, {len(todo)} utterances')

        print('Extracting acoustic feature...', flush=True)
        x = Parallel(n_jobs=args.n_jobs)(delayed(extract_feature)(
            str(file), feature=args.feature_type, cmvn=args.apply_cmvn)
                                         for file in tqdm(todo))

        print('Encoding target...', flush=True)
        y_phones_boundaries = Parallel(n_jobs=args.n_jobs)(
            delayed(read_text)(str(file), target=args.target)
            for file in tqdm(todo))
        y_phones, y_boundaries = list(zip(* y_phones_boundaries))
        y_large, _ = encode_target(y_phones,
                                   table=encode_table,
                                   mode=args.target,
                                   max_idx=args.n_tokens)

        def rescale_boundary(boundary, feature_seqlen):
            # Because orginal boundaries are based on waveform, here we
            # extract features and thus need to rescale the boundaries
            # upon extracted features
            original_seqlen = boundary[-1] - boundary[0]
            rescale_factor = feature_seqlen / original_seqlen
            scaled_boundary = np.array(boundary) * rescale_factor
            rounded_boundary = list(scaled_boundary.round().astype(np.int64))
            assert rounded_boundary[-1] == feature_seqlen
            return rounded_boundary

        scaled_boundaries = [rescale_boundary(boundary, feature.shape[0]) for boundary, feature in zip(y_boundaries, x)]

        def convert_labels(labels, id2id):
            converted = []
            for label in labels:
                converted.append([])
                for phoneid in label:
                    converted[-1].append(id2id[phoneid])
            return converted

        y_medium = convert_labels(y_large, phonesets['large2medium']['id'])
        y_small = convert_labels(y_large, phonesets['large2small']['id'])

        paths = [truncate_path(path, mode) for path in todo]
        df = pd.DataFrame(
            data={
                'fileid': paths,
                'featureid': np.arange(len(x)),
                'feature_seqlen': [len(item) for item in x],
                'feature_boundary': scaled_boundaries,
                'phoneid_large': y_large,
                'phoneid_medium': y_medium,
                'phoneid_small': y_small,
                'phoneid_seqlen': [len(item) for item in y_large],
            })
        df = df.sort_values(by=['feature_seqlen', 'phoneid_seqlen'])

        x_dict = {}
        for idx, row in df.iterrows():
            featureid = row['featureid']
            feature = x[featureid]
            x_dict[featureid] = feature
        # save accoustic feature
        with open(os.path.join(output_dir, f'{mode}.pkl'), 'wb') as handle:
            pickle.dump(x_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save phoneme label sequence
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(output_dir, f'{mode}.csv'), index=False)

        # save phoneme-id mapping
        with open(os.path.join(output_dir, 'phonesets.pkl'), 'wb') as handle:
            pickle.dump(phonesets, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_phonesets(phonemap_path):
    phonesets = {
        'large': {},
        'medium': {},
        'small': {},
        'large2medium': {},
        'medium2small': {},
        'large2small': {}
    }
    for name in list(phonesets.keys())[:3]:
        phonesets[name]['phone2id'] = {'<sos>': 0, '<eos>': 1}
        phonesets[name]['id2phone'] = {0: '<sos>', 1: '<eos>'}

    for name in list(phonesets.keys())[3:]:
        phonesets[name]['phone'] = {
            '<sos>': '<sos>',
            '<eos>': '<eos>',
        }
        phonesets[name]['id'] = {0: 0, 1: 1}

    # phone2id cache
    large = phonesets['large']['phone2id']
    medium = phonesets['medium']['phone2id']
    small = phonesets['small']['phone2id']

    for line in open(phonemap_path, 'r').readlines():
        phones = line[:-1].split('\t')
        for phoneset_name, phone in zip(['large', 'medium', 'small'], phones):
            phoneset = phonesets[phoneset_name]
            if phone not in phoneset['phone2id'].keys():
                phone_id = len(phoneset['phone2id'].keys())
                phoneset['phone2id'][phone] = phone_id
                phoneset['id2phone'][phone_id] = phone

        phonesets['large2medium']['phone'][phones[0]] = phones[1]
        phonesets['large2small']['phone'][phones[0]] = phones[2]
        phonesets['medium2small']['phone'][phones[1]] = phones[2]
        phonesets['large2medium']['id'][large[phones[0]]] = medium[phones[1]]
        phonesets['large2small']['id'][large[phones[0]]] = small[phones[2]]
        phonesets['medium2small']['id'][medium[phones[1]]] = small[phones[2]]

    return phonesets


def get_preprocess_args():

    parser = argparse.ArgumentParser(
        description='preprocess arguments for Timit dataset.')

    parser.add_argument('--data_path',
                        default='/home/leo/d/datasets/timit',
                        type=str,
                        help='Path to raw TIMIT dataset')
    parser.add_argument('--phonemap_path',
                        default='./utility/phones.60-48-39.map.txt',
                        type=str,
                        help='Path to kaldi phone mapping textfile')
    parser.add_argument('--testspk_path',
                        default='./utility/test_spk.list',
                        type=str,
                        help='Path to kaldi phone mapping textfile')
    parser.add_argument('--output_path',
                        default='./data/',
                        type=str,
                        help='Path to store output',
                        required=False)
    parser.add_argument('--feature_type',
                        default='mel',
                        type=str,
                        help='Feature type ( mfcc / fbank / mel / linear )',
                        required=False)
    parser.add_argument('--apply_cmvn',
                        default=True,
                        type=boolean_string,
                        help='Apply CMVN on feature',
                        required=False)
    parser.add_argument('--n_jobs',
                        default=-1,
                        type=int,
                        help='Number of jobs used for feature extraction',
                        required=False)
    parser.add_argument('--n_tokens',
                        default=1000,
                        type=int,
                        help='Vocabulary size of target',
                        required=False)
    parser.add_argument(
        '--target',
        default='phoneme',
        type=str,
        help='Learning target ( phoneme / char / subword / word )',
        required=False)

    args = parser.parse_args()
    return args


def main():
    # get arguments
    args = get_preprocess_args()
    dim = num_freq if args.feature_type == 'linear' else mel_dim

    # get phoneset
    phonesets = get_phonesets(args.phonemap_path)

    # process data
    preprocess(args, dim, phonesets)


if __name__ == '__main__':
    main()
