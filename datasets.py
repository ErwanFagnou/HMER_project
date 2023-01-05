import bz2
import os.path
import pickle
from typing import Dict

import itertools

import torch
from torch.utils.data import DataLoader

import config


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
        imgs = {k: torch.from_numpy(v) for k, v in imgs.items()}
        return imgs


def read_labels(file):
    with open(file, 'r') as f:
        lines = f.read().splitlines()
    lines = [line.split() for line in lines]
    return {line[0]: line[1:] for line in lines}


def preprocess_image(img, max_width, max_height, downscale=1):
    if downscale > 1:
        img = torch.nn.functional.avg_pool2d(img[None].type(torch.LongTensor), downscale)[0]

    if max_width is not None:
        h, w = img.shape
        padded_img = torch.zeros(max_height, max_width, dtype=img.dtype)
        padded_img[:h, :w] = img  # [:max_height, :max_width]
        img = padded_img
    return img


def preprocess_labels(labels, label2id, max_len=None, include_sos_and_eos=False):
    """ Label preprocessing: tokenize and pad """
    if include_sos_and_eos:
        labels = ["<sos>"] + labels + ["<eos>"]
    if max_len is not None:
        labels = labels[:max_len] + ["<pad>"] * (max_len - len(labels))
    return torch.Tensor([label2id[label] for label in labels])


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


def get_padding_collate_fn(label2id, max_img_h, max_img_w, max_label_len):
    pad_id = label2id['<pad>']
    def padding_collate_fn(batch):
        max_img_w = max([item[0].shape[1] for item in batch])
        max_img_h = max([item[0].shape[0] for item in batch])
        max_label_len = max([len(item[1]) for item in batch])

        images = torch.stack([preprocess_image(item[0], max_img_w, max_img_h) for item in batch])
        labels = torch.stack([torch.cat((item[1], torch.tensor([pad_id]*(max_label_len-len(item[1])), dtype=item[1].dtype))) for item in batch])
        return images, labels
    return padding_collate_fn


class DatasetManager:
    def __init__(self, data_files: DatasetFiles, batch_size, precompute_padding=False, batch_padding=False, downscale=1, include_sos_and_eos=False, verbose=True):
        assert not (precompute_padding and batch_padding), "Only one of precompute_padding and batch_padding can be True"

        self.data_files = data_files

        self.label2id = None
        self.train_loader: DataLoader = None
        self.test_loaders: Dict[str, DataLoader] = None

        self.max_img_h = self.max_img_w = self.max_label_len = None

        suffix = '_padded' if precompute_padding else ''
        if downscale > 1:
            suffix += '_downscale=' + str(downscale)
        if include_sos_and_eos:
            suffix += '_sos_and_eos'
        file_path = os.path.join('./data/datasets/', data_files.name + suffix + '.pkl.bz2')

        if os.path.exists(file_path):
            # load the cached datasets
            if verbose:
                print(f'Loading cached data from "{file_path}"...')
            with bz2.BZ2File(file_path, 'rb') as f:
                train_dataset, test_datasets, self.label2id = pickle.load(f)
        else:
            # load and preprocess the datasets
            if verbose:
                print(f'First time loading and preprocessing the dataset...')
            self.label2id = self.load_dictionary()
            train_dataset, test_datasets = self.get_datasets(precompute_padding, downscale, include_sos_and_eos)

            # save the datasets to file
            if precompute_padding:
                if verbose:
                    print('Not saving the dataset: padded images are too large.')
            else:
                if verbose:
                    print(f'Saving dataset to "{file_path} for future use...')
                with bz2.BZ2File(file_path, 'wb') as f:
                    pickle.dump((train_dataset, test_datasets, self.label2id), f)

        self.id2label = {v: k for k, v in self.label2id.items()}

        # to data loaders
        self.make_loaders(train_dataset, test_datasets, batch_size, batch_padding)

    def load_dictionary(self):
        with open(self.data_files.dict, 'r') as f:
            dictionary = f.read().splitlines()
        label2id = config.additional_tokens
        offset = max(label2id.values()) + 1
        label2id.update({label: i + offset for i, label in enumerate(dictionary)})
        return label2id

    def get_datasets(self, precompute_padding=False, downscale=1, include_sos_and_eos=False):
        # load the raw data from files
        train_datasets = read_datasets(self.data_files.train)
        train_dataset = merge_datasets(train_datasets)
        test_datasets = read_datasets(self.data_files.test)

        # preprocess data
        if precompute_padding:
            all_img_shapes = [img.shape for img, _ in
                              itertools.chain(train_dataset.values(), *[d.values() for d in test_datasets.values()])]
            max_img_h = max([s[0] for s in all_img_shapes])
            max_img_w = max([s[1] for s in all_img_shapes])
            max_label_len = max([len(labels) for _, labels in train_dataset.values()]) + 2
        else:
            max_img_h = max_img_w = max_label_len = None

        train_dataset = {
            name: (preprocess_image(img, max_img_w, max_img_h, downscale), preprocess_labels(labels, self.label2id, max_label_len, include_sos_and_eos))
            for name, (img, labels) in train_dataset.items()}
        for name, dataset in test_datasets.items():
            max_label_len = max([len(labels) for _, labels in dataset.values()]) + 2
            test_datasets[name] = {
                name: (preprocess_image(img, max_img_w, max_img_h, downscale), preprocess_labels(labels, self.label2id, max_label_len, include_sos_and_eos))
                for name, (img, labels) in dataset.items()}

        return train_dataset, test_datasets

    def make_loaders(self, train_dataset, test_datasets, batch_size, batch_padding=False):
        all_img_shapes = [img.shape for img, _ in
                          itertools.chain(train_dataset.values(), *[d.values() for d in test_datasets.values()])]
        self.max_img_h = max([s[0] for s in all_img_shapes])
        self.max_img_w = max([s[1] for s in all_img_shapes])
        self.max_label_len = max([len(labels) for _, labels in train_dataset.values()]) + 2

        if batch_padding:
            collate_fn = get_padding_collate_fn(self.label2id, self.max_img_h, self.max_img_w, self.max_label_len)
        else:
            collate_fn = multilength_collate_fn

        self.train_loader = DataLoader(list(train_dataset.values()), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.test_loaders = {name: DataLoader(list(dataset.values()), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                             for name, dataset in test_datasets.items()}


if __name__ == '__main__':
    crohme = DatasetManager(CROHME, batch_size=32)

    print('Train samples:', len(crohme.train_loader.dataset))
    print('Test samples:', {name: len(loader.dataset) for name, loader in crohme.test_loaders.items()})

    import matplotlib.pyplot as plt
    for batch in crohme.train_loader:
        print(len(batch[0]), batch[0][0].shape)
        print(len(batch[1]), batch[1][0].shape)
        plt.imshow(batch[0][0])
        plt.show()
        # break
