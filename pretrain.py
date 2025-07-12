import os
import time
import wandb
import warnings
import argparse
from tqdm import tqdm
from datasets.dataset import balance_dataloader
from datasets.dataset import imbalance_dataloader

import torch
import torch.nn as nn
import torch.amp as amp
import torch.optim as optim
import torch.nn.functional as F

from loss  import *
from utils import *
from models.resnet import resnet50, resnet50_fe
from models.resnet import resnet101, resnet101_fe
from models.resnet import resnet152, resnet152_fe

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Fine-Tuning ResNet-101 on CIFAR10')

parser.add_argument('--net', default='ResNet-101')
parser.add_argument('--opt', default='SGD')
parser.add_argument('--dataset', type=str, default='CIFAR10')

parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--batch_size',  type=int, default=256)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--imb_factor',  type=float, default=0.01)
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')

parser.add_argument('--mixup',  type=bool, default=False)
parser.add_argument('--longtail', type=bool, default=True)
parser.add_argument('--finetune', type=bool, default=True)
args = parser.parse_args()

warnings.filterwarnings("ignore")

# register weight and bias
name = f"({('LongTail(' + str(args.imb_factor) + ')' if args.longtail else 'Balance')}{(',mixup' if args.mixup else '')})"
wandb.init(project="ImBalance(KBLS)", entity="scut_zeno", name="{}{}_FineTune_{}".format(name, args.net, args.dataset))
wandb.config.update(args)


print(f'==> Preparing {args.dataset} Dataset ...\n')
if args.longtail:
    train_loader, valid_loader, cls_num_list, n_classes = imbalance_dataloader(args.dataset, args.batch_size, args.imb_factor)
else:
    train_loader, valid_loader, n_classes = balance_dataloader(args.dataset, args.batch_size)


# ----------------------------------------------------
# fine-tuning the pretrained model on current dataset
# ----------------------------------------------------
if args.finetune:
    # set ResNet Model
    print(f'==> Building ResNet BackBone Net ...')
    if args.net == 'ResNet-50':    
        net = resnet50(pretrained=True,  progress=True, num_classes=n_classes).to(device)
    elif args.net == 'ResNet-101': 
        net = resnet101(pretrained=True, progress=True, num_classes=n_classes).to(device)
    elif args.net == 'ResNet-152': 
        net = resnet152(pretrained=True, progress=True, num_classes=n_classes).to(device)
    else: 
        print(f"Error: '{args.net}' model is not supported...\n")
        sys.exit(1)
    # calculate the number of parameters for this model
    parameter_count = sum(p.numel() for p in net.parameters())
    print(f"+--------------------------------------------------------+")
    print(f"|-- {args.net} BackBone NetWork has {parameter_count / 1e6:.2f}M parameters. --|")
    print(f"+--------------------------------------------------------+\n")

    print(f'==> Fine-tuning {args.net} on {args.dataset} ...')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    scaler = amp.GradScaler()  # auto gradient scaling
    optimizer = optim.SGD([{'params': net.parameters()}], args.lr, momentum=0.9, weight_decay=2e-4)
    scheduler = CosineAnnealingLRWarmup(
                    optimizer=optimizer, 
                    T_max=args.n_epochs,
                    eta_min=1e-5,
                    warmup_epochs=args.warmup_epochs,
                    base_lr=args.lr / args.warmup_epochs,
                    warmup_lr=args.lr)

    best_acc = .0  # store best accuracy during network finetuning
    record_time = time.ctime()  # keep record of the starting time

    for epoch in range(args.n_epochs):
        start = time.time()

        # train and valid
        train_loss = train(net, device, train_loader, scaler, criterion, optimizer, epoch, mixup=args.mixup)
        valid_loss, acc = valid(net, device, valid_loader, criterion)
        
        scheduler.step(epoch)  # update learning rate

        # store the best fine-tuned model for feature extraction
        if acc > best_acc:
            print('Saving...')
            torch.save(net.state_dict(), f'./checkpoints/{args.net}_{args.dataset}{name}.pth')
            best_acc = acc

        content = time.ctime() + ' ' + (f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.8f}, '
                                    f'val loss: {valid_loss:.5f}, acc: {(acc):.3f}')
        print(content)
        os.makedirs(f"log/{args.net}_{args.dataset}", exist_ok=True)
        with open('log/[{}]{}.txt'.format(record_time, name), 'a') as appender:
            appender.write(content + "\n")

        # update the train process to wandb
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'eval_loss': valid_loss, "val_acc": acc,
               "lr": optimizer.param_groups[0]["lr"], "epoch_time": time.time() - start})

    print(f'=== finish ===\n')
    wandb.save("wandb_{}_FineTune_{}.h".format(args.net, args.dataset))


# -----------------------------------------
# extract features of the data from ResNet
# -----------------------------------------
print(f'==> Extracting features from ResNet BackBone Network ...')

net = resnet50_fe(pretrained=False, progress=True).to(device)
checkpoint = torch.load(f'./checkpoints/{args.net}_{args.dataset}{name}.pth')
net.load_state_dict(checkpoint, strict=False)

train_loader, valid_loader, n_classes = balance_dataloader(args.dataset, batch_size=2000, augment=False)

train_feature_dataset, valid_feature_dataset = extract_features(net, train_loader, valid_loader, device)
    
# store the features and labels  ./datasets/features/{}
save_dir = "./datasets/features/{}".format(args.dataset)
os.makedirs(save_dir, exist_ok=True)
torch.save(train_feature_dataset, os.path.join(save_dir, f"train({args.net}).pt"))
torch.save(valid_feature_dataset, os.path.join(save_dir, f"valid({args.net}).pt"))

print(f'== finish ==\n')
