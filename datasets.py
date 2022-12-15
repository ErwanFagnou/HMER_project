import bz2
import os.path
import pickle
from typing import Dict

import numpy as np

import itertools

from torch.utils.data import DataLoader


class DatasetFiles:
    def __init__(self, name, train, test, dict):
        self.name = name
        self.train = train
        self.test = test
        self.dict = dict


CROHME = DatasetFiles(
    name='CROHME',
    train=[
        ('TRAIN14', './data/datasets/CROHME/2014_train_images.pkl.bz2', './data/datasets/CROHME/2014_train_labels.txt')
    ],
    test=[
        ('TEST14', './data/datasets/CROHME/2014_test_images.pkl.bz2', './data/datasets/CROHME/2014_test_labels.txt'),
        ('TEST16', './data/datasets/CROHME/2016_test_images.pkl.bz2', './data/datasets/CROHME/2016_test_labels.txt'),
        ('TEST19', './data/datasets/CROHME/2019_test_images.pkl.bz2', './data/datasets/CROHME/2019_test_labels.txt'),
    ],
    dict='./data/datasets/CROHME/dict.txt'
)


def read_images(file):
    with bz2.BZ2File(file, 'rb') as f:
        imgs = pickle.load(f)
        imgs = {k: (v > 0) for k, v in imgs.items()}
        return imgs


def read_labels(file):
    with open(file, 'r') as f:
        lines = f.read().splitlines()
    lines = [line.split() for line in lines]
    return {line[0]: line[1:] for line in lines}


def preprocess_image(img, max_width, max_height):
    h, w = img.shape
    padded_img = np.zeros((max_height, max_width), dtype=img.dtype)
    padded_img[:h, :w] = img  # [:max_height, :max_width]
    return padded_img


def preprocess_labels(labels, label2id, max_len=None):
    """ Label preprocessing: tokenize and pad """
    labels = ["<sos>"] + labels + ["<eos>"]
    if max_len is not None:
        labels = labels[:max_len] + ["<pad>"] * (max_len - len(labels))
    return np.array([label2id[label] for label in labels])


def read_datasets(files):
    """ Reads multiple datasets separately """
    datasets = {}
    for i, (name, x_file, y_file) in enumerate(files):
        images = read_images(x_file)
        labels = read_labels(y_file)
        dataset = {k: (images[k], labels[k]) for k in images.keys()}
        datasets[name] = dataset
    return datasets


def merge_datasets(datasets):
    """ Merges multiple datasets """
    merged = {}
    for i, (name, dataset) in enumerate(datasets.items()):
        prefix = name + '_'
        file_dict = {prefix + k: v for k, v in dataset.items()}
        merged.update(file_dict)
    return merged


def multilength_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


class DatasetManager:
    def __init__(self, data_files: DatasetFiles, batch_size, padding=False, verbose=True):
        self.data_files = data_files

        self.label2id = None
        self.train_loader: DataLoader = None
        self.test_loaders: Dict[DataLoader] = None

        suffix = '_padded' if padding else ''
        file_path = os.path.join('./data/datasets/', data_files.name + suffix + '.pkl.bz2')

        if os.path.exists(file_path):
            # load the cached datasets
            if verbose:
                print(f'Loading cached data from "{file_path}"...')
            with bz2.BZ2File(file_path, 'rb') as f:
                train_dataset, test_datasets, self.label2id = pickle.load(f)
                self.make_loaders(train_dataset, test_datasets, batch_size)
        else:
            # load and preprocess the datasets
            if verbose:
                print(f'First time loading and preprocessing the dataset...')
            self.label2id = self.load_dict()
            train_dataset, test_datasets = self.get_datasets(padding)

            # save the datasets to file
            if padding:
                if verbose:
                    print('Not saving the dataset: padded images are too large.')
            else:
                if verbose:
                    print(f'Saving dataset to "{file_path} for future use...')
                with bz2.BZ2File(file_path, 'wb') as f:
                    pickle.dump((train_dataset, test_datasets, self.label2id), f)

        # to data loaders
        self.make_loaders(train_dataset, test_datasets, batch_size)

    def load_dict(self):
        with open(self.data_files.dict, 'r') as f:
            dict = f.read().splitlines()
        label2id = {'<sos>': 0, '<eos>': 1, '<pad>': 2}
        label2id.update({label: i + 2 for i, label in enumerate(dict)})
        return label2id

    def get_datasets(self, padding=False):
        # load the raw data from files
        train_datasets = read_datasets(self.data_files.train)
        train_dataset = merge_datasets(train_datasets)
        test_datasets = read_datasets(self.data_files.test)

        # preprocess data
        if padding:
            all_img_shapes = [img.shape for img, _ in
                              itertools.chain(train_dataset.values(), *[d.values() for d in test_datasets.values()])]
            max_img_h = max([s[0] for s in all_img_shapes])
            max_img_w = max([s[1] for s in all_img_shapes])
            max_label_len = max([len(labels) for _, labels in train_dataset.values()]) + 2

            train_dataset = {
                name: (preprocess_image(img, max_img_w, max_img_h), preprocess_labels(labels, self.label2id, max_label_len))
                for name, (img, labels) in train_dataset.items()}
            for name, dataset in test_datasets.items():
                max_label_len = max([len(labels) for _, labels in dataset.values()]) + 2
                test_datasets[name] = {
                    name: (preprocess_image(img, max_img_w, max_img_h), preprocess_labels(labels, self.label2id, max_label_len))
                    for name, (img, labels) in dataset.items()}

        else:
            train_dataset = {name: (img, preprocess_labels(labels, self.label2id, None))
                             for name, (img, labels) in train_dataset.items()}
            for name, dataset in test_datasets.items():
                test_datasets[name] = {name: (img, preprocess_labels(labels, self.label2id, None))
                                       for name, (img, labels) in dataset.items()}

        return train_dataset, test_datasets

    def make_loaders(self, train_dataset, test_datasets, batch_size):
        self.train_loader = DataLoader(list(train_dataset.values()), batch_size=batch_size, shuffle=True, collate_fn=multilength_collate_fn)
        self.test_loaders = {name: DataLoader(list(dataset.values()), batch_size=batch_size, shuffle=False, collate_fn=multilength_collate_fn)
                             for name, dataset in test_datasets.items()}


if __name__ == '__main__':
    crohme = DatasetManager(CROHME, batch_size=32, padding=False)

    print('Train samples:', len(crohme.train_loader.dataset))
    print('Test samples:', {name: len(loader.dataset) for name, loader in crohme.test_loaders.items()})
