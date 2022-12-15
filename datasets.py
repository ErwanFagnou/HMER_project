import bz2
import pickle

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
        ('train14', './data/datasets/CROHME/2014_train_images.pkl.bz2', './data/datasets/CROHME/2014_train_labels.txt')
    ],
    test=[
        ('test14', './data/datasets/CROHME/2014_test_images.pkl.bz2', './data/datasets/CROHME/2014_test_labels.txt'),
        ('test16', './data/datasets/CROHME/2016_test_images.pkl.bz2', './data/datasets/CROHME/2016_test_labels.txt'),
        ('test19', './data/datasets/CROHME/2019_test_images.pkl.bz2', './data/datasets/CROHME/2019_test_labels.txt'),
    ],
    dict='./data/datasets/CROHME/dict.txt'
)


def read_images(file):
    with bz2.BZ2File(file, 'rb') as f:
        return pickle.load(f)


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



class DatasetManager:
    def __init__(self, data_files: DatasetFiles, batch_size):
        # todo: cache the datasets
        self.data_files = data_files

        self.train_loader = None
        self.test_loaders = None

        self.label2id = self.load_dict()
        self.load_datasets(batch_size)

    def load_dict(self):
        with open(self.data_files.dict, 'r') as f:
            dict = f.read().splitlines()
        label2id = {'<sos>': 0, '<eos>': 1, '<pad>': 2}
        label2id.update({label: i + 2 for i, label in enumerate(dict)})
        return label2id

    def load_datasets(self, batch_size):
        # load the raw data from files
        train_datasets = read_datasets(self.data_files.train)
        train_dataset = merge_datasets(train_datasets)
        test_datasets = read_datasets(self.data_files.test)

        # preprocess data (no padding for test labels)
        all_img_shapes = [img.shape for img, _ in
                          itertools.chain(train_dataset.values(), *[d.values() for d in test_datasets.values()])]
        max_img_h = max([s[0] for s in all_img_shapes])
        max_img_w = max([s[1] for s in all_img_shapes])
        max_label_len = max([len(labels) for _, labels in train_dataset.values()]) + 2

        train_dataset = {
            name: (preprocess_image(img, max_img_w, max_img_h), preprocess_labels(labels, self.label2id, max_label_len))
            for name, (img, labels) in train_dataset.items()}
        for name, dataset in test_datasets.items():
            test_datasets[name] = {
                name: (preprocess_image(img, max_img_w, max_img_h), preprocess_labels(labels, self.label2id, None))
                for name, (img, labels) in dataset.items()}

        # to data loaders
        self.train_loader = DataLoader(list(train_dataset.values()), batch_size=batch_size, shuffle=True)
        self.test_loaders = {name: DataLoader(list(dataset.values()), batch_size=batch_size, shuffle=False)
                             for name, dataset in test_datasets.items()}


if __name__ == '__main__':
    crohme = DatasetManager(CROHME, batch_size=32)

    print('Train:')
    for i, (img, label) in enumerate(crohme.train_loader):
        print(img.shape, len(label.shape))
        if i >= 3:
            break
