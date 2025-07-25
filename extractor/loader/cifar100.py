import random
import numpy as np
from .sampler import ClassAwareSampler

import torch
import torchvision
from torchvision import transforms
import torchvision.datasets

from PIL import Image


class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, config, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.train = train
        self.config = config
        # get the number of images per class
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        # generate imbalanced data
        self.gen_imbalanced_data(img_num_list)
        # 
        self.dual_sample = config.sampler.dual_sample
        if self.dual_sample:
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
            self.class_dict = self._get_class_dict()

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        ''' get the number of images per class '''
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def sample_class_index_by_weight(self):
        ''' sample a class index by weight '''
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def get_num_classes(self):
        return self.cls_num

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def get_annotations(self):
        ''' get the annotations of the dataset '''
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    def get_cls_num_list(self):
        ''' get the number of images per class '''
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def _get_class_dict(self):
        ''' get the class dictionary of the dataset '''
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        ''' get the weight of each class '''
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, index):
        ''' get the item at the given index '''
        # Return: (image, target) where the target is index of the target class.
        
        meta = dict()
        if self.config.sampler.dual_sample.enable: #balance sampler
            assert self.config.sampler.dual_sample.type in ["balance", "reverse", "long-tailed"]
            if self.config.sampler.dual_sample.type == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.config.sampler.dual_sample.type == "balance":
                sample_class = random.randint(0, self.cls_num-1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.config.sampler.dual_sample.type == "long-tailed":
                sample_index = index #random.randint(0, self.__len__() - 1)
            
            sample_image = self.data[sample_index]
            sample_label = self.targets[sample_index]
            sample_image = Image.fromarray(sample_image)
            sample_image = self.transform(sample_image)

            meta['sample_image'] = sample_image
            meta['sample_label'] = sample_label

        if self.config.sampler.type == "weighted sampler" and self.train:
            assert self.config.sampler.weighted_sampler.type in ["balance", "reverse", "long-tailed"]
            if  self.config.sampler.weighted_sampler.type == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
            elif self.config.sampler.weighted_sampler.type == "reverse":
                sample_class = self.sample_class_index_by_weight()
            elif self.config.sampler.dual_sample.type == "long-tailed":
                sample_index = index 
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)  # return a PIL Image for consistency with other datasets

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target  # , meta


    def gen_imbalanced_data(self, img_num_per_cls):
        '''  '''
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets



class CIFAR100_LT(object):
    def __init__(self, config, distributed, root='../datasets', imb_type='exp',
                    imb_factor=0.01, batch_size=128, num_works=40, dual_sample=False):
        # define the transform for training and evaluation
        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        
        train_dataset = IMBALANCECIFAR100(config, root=root, imb_type=imb_type, 
                                          imb_factor=imb_factor, rand_number=0, 
                                          train=True, download=True, transform=train_transform)
        
        eval_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)
        
        self.cls_num_list = train_dataset.get_cls_num_list()

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler) ##drop_last=True

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler) 

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)