import os
import sys
import time
import wandb
import random
import argparse
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.utils.data import DataLoader
from dataloader.dataset import data_loader
from torch.utils.data import TensorDataset
from utils.features_utils import extract_features

from models.resnet import FFN
from Experiment.KBLS.models.kbls import KBLS
from Experiment.KBLS.models.bls import BLS


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

warnings.filterwarnings("ignore")

# add parameters
parser = argparse.ArgumentParser(description='Imbalance Learning with KBLS as classifier')
parser.add_argument('--net', default='ResNet-50')
parser.add_argument('--dataset', type=str, default='CIFAR10')

parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--classifier', type=str, default='FFN')
parser.add_argument('--batch_size',  type=int, default=32)
args = parser.parse_args()


# initialize wandb
wandb.init(project="ImBalance(KBLS)", entity="scut_zeno", name="{}_{}_Classifier".format(args.net, args.dataset))
wandb.config.update(args)


# load the feature dataset directly
save_dir = "./datasets/features/{}".format(args.dataset)
train_feature_dataset = torch.load(os.path.join(save_dir, f"{args.net}_train_features.pt"), weights_only=False)
test_feature_dataset  = torch.load(os.path.join(save_dir, f"{args.net}_test_features.pt"),  weights_only=False)

# split the dataset into features and labels
train_features, train_labels = train_feature_dataset.tensors
test_features,  test_labels  = test_feature_dataset.tensors

feature_size = train_features.shape[-1]

# use BLS as classifier:
if args.classifier == 'BLS':
    print("==> Building the KBLS Classifier...")
    classifier = BLS(
        featuretimes=16,
        enhancetimes=8,
        n_classes=10,
        map_function='relu',
        enhance_function='tanh',
        featuresize='auto',
        reg=0.001,
        sig=0.01,
        use_sparse=True
    ).to(device)
    print(f'=== finish ===\n')

    # set KBLS Classifier and fit the training data
    print(f'==> Fitting the training data for BLS classifier ...')
    for epoch in range(args.n_epochs):
        classifier.fit(test_features, test_labels)
    print(f'=== finish ===\n')

    # calculate the validation accuracy
    print(f'==> Test the effectiveness of BLS classifier ...')
    test_outputs = classifier(test_features)
    print(f'=== finish ===\n')

    predicted_labels = torch.argmax(test_outputs, dim=1)

    acc = (predicted_labels == test_labels.to(device)).sum().item() / len(test_labels)
    print("evaluation accuracy: {:.3f}%".format(acc * 100))


if args.classifier == 'FFN':
    print("==> Building the FFN Classifier...")
    classifier = FFN(
        feat_in = feature_size,
        n_classes = 100
    ).to(device)

    # 
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=args.batch_size,
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, args.n_epochs)

    print("==> Training the FFN classifier...")

    for epoch in range(args.n_epochs):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{args.n_epochs}',
            unit='batch'
        )
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条描述
            progress_bar.set_postfix({
                'Loss': running_loss/(total/inputs.size(0)),
                'Acc': 100*correct/total
            })
        
        # 每个epoch的统计结果
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

        # evaluate the model
        classifier.eval()
        eval_correct = 0
        eval_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = classifier(inputs)
                _, predicted = torch.max(outputs.data, 1)
                eval_total += labels.size(0)
                eval_correct += (predicted == labels).sum().item()
                
        eval_acc = 100 * eval_correct / eval_total
        print(f'Epoch {epoch+1} - Eval Acc: {eval_acc:.2f}%')

        wandb.log({'epoch': epoch, 'train_loss': epoch_loss, 'train_acc': epoch_acc, 'test_acc': eval_acc})

    print('== finish ==')

wandb.save("{}_{}_Classifier.h".format(args.net, args.dataset))