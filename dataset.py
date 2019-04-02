from io import BytesIO
import pickle
import PIL
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, RandomSampler, BatchSampler, _int_classes
from numpy.random import choice
import csv
from copy import copy
import codecs
from torch._utils import _accumulate
from collections import Counter


class RandomSamplerReplacment(torch.utils.data.sampler.Sampler):
    """Samples elements randomly, with replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(torch.from_numpy(choice(self.num_samples, self.num_samples, replace=True)))

    def __len__(self):
        return self.num_samples


class LimitDataset(Dataset):

    def __init__(self, dset, max_len):
        self.dset = dset
        self.max_len = max_len

    def __len__(self):
        return min(len(self.dset), self.max_len)

    def __getitem__(self, index):
        return self.dset[index]


class ByClassDataset(Dataset):

    def __init__(self, ds):
        self.dataset = ds
        self.idx_by_class = {}
        for idx, (_, c) in enumerate(ds):
            self.idx_by_class.setdefault(c, [])
            self.idx_by_class[c].append(idx)

    def __len__(self):
        return min([len(d) for d in self.idx_by_class.values()])

    def __getitem__(self, idx):
        idx_per_class = [self.idx_by_class[c][idx]
                         for c in range(len(self.idx_by_class))]
        labels = torch.LongTensor([self.dataset[i][1]
                                   for i in idx_per_class])
        items = [self.dataset[i][0] for i in idx_per_class]
        if torch.is_tensor(items[0]):
            items = torch.stack(items)

        return (items, labels)


class IdxDataset(Dataset):
    """docstring for IdxDataset."""

    def __init__(self, dset):
        super(IdxDataset, self).__init__()
        self.dset = dset
        self.idxs = range(len(self.dset))

    def __getitem__(self, idx):
        data, labels = self.dset[self.idxs[idx]]
        return (idx, data, labels)

    def __len__(self):
        return len(self.idxs)


def image_loader(imagebytes):
    img = PIL.Image.open(BytesIO(imagebytes))
    return img.convert('RGB')


class IndexedFileDataset(Dataset):
    """ A dataset that consists of an indexed file (with sample offsets in
        another file). For example, a .tar that contains image files.
        The dataset does not extract the samples, but works with the indexed
        file directly.
        NOTE: The index file is assumed to be a pickled list of 3-tuples:
        (name, offset, size).
    """

    def __init__(self, filename, index_filename=None, extract_target_fn=None,
                 transform=None, target_transform=None, loader=image_loader):
        super(IndexedFileDataset, self).__init__()

        # Defaults
        if index_filename is None:
            index_filename = filename + '.index'
        if extract_target_fn is None:
            extract_target_fn = lambda *args: args

        # Read index
        with open(index_filename, 'rb') as index_fp:
            sample_list = pickle.load(index_fp)

        # Collect unique targets (sorted by name)
        targetset = set(extract_target_fn(target)
                        for target, _, _ in sample_list)
        targetmap = {target: i for i, target in enumerate(sorted(targetset))}

        self.samples = [(targetmap[extract_target_fn(target)], offset, size)
                        for target, offset, size in sample_list]
        self.filename = filename

        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def _get_sample(self, fp, idx):
        target, offset, size = self.samples[idx]
        fp.seek(offset)
        sample = self.loader(fp.read(size))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __getitem__(self, index):
        with open(self.filename, 'rb') as fp:
            # Handle slices
            if isinstance(index, slice):
                return [self._get_sample(fp, subidx) for subidx in
                        range(index.start or 0, index.stop or len(self),
                              index.step or 1)]

            return self._get_sample(fp, index)

    def __len__(self):
        return len(self.samples)


class DuplicateBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, duplicates, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.duplicates = duplicates

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch * self.duplicates
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch * self.duplicates

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def list_line_locations(filename, limit=None):
    line_offset = []
    offset = 0
    with open(filename, "rb") as f:
        for line in f:
            line_offset.append(offset)
            offset += len(line)
            if limit is not None and len(line_offset) > limit:
                break
    return line_offset


def _load_or_create(filename, create_fn, cache=True, force_create=False):
    loaded = False
    if not force_create:
        try:
            with open(filename, 'rb') as fp:
                value = pickle.load(fp)
            loaded = True
        except:
            pass
    if not loaded:
        value = create_fn()
    if cache and not loaded:
        with open(filename, 'wb') as fp:
            pickle.dump(value, fp)
    return value


class LinedTextDataset(Dataset):
    """ Dataset in which every line is a seperate item (e.g translation)
    """

    def __init__(self, filename, transform=None, cache=True):
        self.filename = filename
        self.transform = transform
        self.items = _load_or_create(filename + '_cached_lines',
                                     create_fn=lambda: list_line_locations(
                                         filename),
                                     cache=cache)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
        with codecs.open(self.filename, encoding='UTF-8') as f:
            f.seek(self.items[index])
            item = f.readline()
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.items)

    def select_range(self, start, end):
        new_dataset = copy(self)
        new_dataset.items = new_dataset.items[start:end]
        return new_dataset

    def filter(self, filter_func):
        new_dataset = copy(self)
        new_dataset.items = [item for item in self if filter_func(item)]
        return new_dataset

    def subset(self, indices):
        new_dataset = copy(self)
        new_dataset.items = [new_dataset.items[idx] for idx in indices]
        return new_dataset

    def split(self, lengths):
        """
        split a dataset into non-overlapping new datasets of given lengths.
        Arguments:
            dataset (Dataset): Dataset to be split
            lengths (sequence): lengths of splits to be produced
        """
        if sum(lengths) != len(self):
            raise ValueError(
                "Sum of input lengths does not equal the length of the input dataset!")

        return [self.select_range(offset-length, offset) for offset, length in zip(_accumulate(lengths), lengths)]

    def random_split(self, lengths):
        """
        Randomly split a dataset into non-overlapping new datasets of given lengths.
        Arguments:
            dataset (Dataset): Dataset to be split
            lengths (sequence): lengths of splits to be produced
        """
        if sum(lengths) != len(self):
            raise ValueError(
                "Sum of input lengths does not equal the length of the input dataset!")

        indices = torch.randperm(sum(lengths)).tolist()
        return [self.subset(indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class CSVDataset(LinedTextDataset):
    """ Dataset with delimited items and pre-knwon fieldnames (no header)
    """

    def __init__(self, filename, fieldnames=None, delimiter='\t', transform=None, cache=True):
        self.filename = filename
        self.fieldnames = fieldnames
        self.delimiter = delimiter
        self.transform = transform
        self.items = _load_or_create(filename + '_cached_lines',
                                     create_fn=lambda: list_line_locations(
                                         filename),
                                     cache=cache)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
        with codecs.open(self.filename, encoding='UTF-8') as f:
            f.seek(self.items[index])
            item = f.readline()
        item = next(csv.DictReader([item],
                                   fieldnames=self.fieldnames,
                                   delimiter=self.delimiter))
        if self.transform is not None:
            item = self.transform(item)
        return item

    def count_fields(self, fieldnames=None):
        fieldnames = fieldnames or self.fieldnames
        counters = {name: Counter() for name in fieldnames}
        for i in range(len(self)):
            value = self[i]
            for field in fieldnames:
                counters[field][value[field]] += 1
        return counters
