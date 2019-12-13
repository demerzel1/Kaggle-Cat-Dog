import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
from model import AlexNet
from dataset import CatDogDataset
from torch.utils.tensorboard import SummaryWriter


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)     


def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu):
    writer = SummaryWriter()
    step = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        begin_time = time.time()

        
        model.train()
        
        train_loss = 0
        train_acc = 0
        count_batch = 0
        
        for data in train_dataloader:
            step += 1
            count_batch += 1

            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            # print(outputs)
            pred = torch.argmax(outputs, dim=1)
            # print(pred)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += torch.sum(pred == labels).to(torch.float32)

            if count_batch % 10 == 0:
                batch_loss = train_loss / (batch_size * count_batch)
                batch_acc = train_acc / (batch_size * count_batch)
                print('Epoch: {} Batch: {} Loss: {} Acc:{} Time: {}'.format(epoch, count_batch, batch_loss, batch_acc, time.time() - begin_time))
                begin_time = time.time()
                writer.add_scalar('Loss/train', batch_loss, step)
                writer.add_scalar('Acc/train', batch_acc, step)
        scheduler.step()
        torch.save(model, './output/epoch_{}.pkl'.format(epoch))
        model.eval()
        for ind, data in enumerate(valid_dataloader):
            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            #print(labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss = loss.item() / batch_size
            pred = torch.argmax(outputs, dim=1)
            valid_acc = torch.sum(pred == labels).to(torch.float32) / batch_size
            print("Valid index: {} Loss: {} Acc: {}".format(ind, valid_loss, valid_acc))

    
if __name__ == '__main__':

    train_data_transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_data_transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    use_gpu = torch.cuda.is_available()

    batch_size = 32
    num_workers = 8

    train_file = '/Users/demerzel/PycharmProjects/cat-dog/data/train.txt'
    valid_file = '/Users/demerzel/PycharmProjects/cat-dog/data/valid.txt'
    test_file = '/Users/demerzel/PycharmProjects/cat-dog/data/test.txt'

    train_dataset = CatDogDataset(file_path=train_file, model='train', data_transforms=train_data_transforms)
    valid_dataset = CatDogDataset(file_path=valid_file, model='train', data_transforms=test_data_transforms)
    test_dataset = CatDogDataset(file_path=test_file, model='test', data_transforms=test_data_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_datasize = len(train_dataset)
    valid_datasize = len(valid_dataset)
    test_datasize = len(test_dataset)

    num_classes = 2
    
    model = AlexNet(num_classes=2)
    model.apply(weights_init)
    '''
    model = models.alexnet(pretrained=True)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    '''
    if use_gpu:
        model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

    model = train_model(model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=exp_lr_scheduler,
                           num_epochs=80,
                           use_gpu=use_gpu)