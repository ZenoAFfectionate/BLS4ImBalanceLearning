import torch
import random

import numpy as np


def get_sampler():
    ''' Choose the sampler you want to use '''
    return ClassAwareSampler
    # return EffectNumSampler
    # return BalancedDatasetSampler


class RandomCycleIter:
    """ Loop through the given data list, randomly shuffle 
        the list after each traversal in non-test mode """
    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
        return self.data_list[self.i]


class BalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """  """
    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices
            
        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples
            
        # calculate the distribution of classes in the dataset
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
        
        per_cls_weights = 1 / np.array(label_to_count)  # weight for each class

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples


class EffectNumberSampler(torch.utils.data.sampler.Sampler):
    """  """
    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices
            
        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    """  """
    i = 0
    j = 0
    while i < n:
        if j >= num_samples_cls: j = 0

        # yield next(data_iter_list[next(cls_iter)])
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1


class ClassAwareSampler(torch.utils.data.sampler.Sampler):
    """  """
    def __init__(self, data_source, num_samples_cls=4, **kwargs):
        num_classes = len(np.unique(data_source.targets))
        self.class_iter = RandomCycleIter(range(num_classes))
        # group sample index by class respectively
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.targets):
            cls_data_list[label].append(i)
        # create a random cycle iterator for each class
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls
        
    def __iter__ (self):
        # 1. choose a class from self.class_iter
        # 2. sample num_samples from the chosen class
        # 3. repeat till the sample number reach self.num_samples_cls
        return class_aware_sample_generator(
            self.class_iter, 
            self.data_iter_list,
            self.num_samples, 
            self.num_samples_cls)
    
    def __len__ (self):
        return self.num_samples
