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
import tqdm

def eval_model(model, use_gpu):
    model.eval()
    results = ''
    ind = 0
    for data in tqdm.tqdm(test_dataloader):
        inputs = data
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        outputs = model(inputs)
        # print(outputs)
        pred = torch.argmax(outputs, dim=1)
        # print(pred)
        for p in pred:
            results = results + str(ind) + ',' + str(p.item()) + '\n'
            ind += 1
    with open('./submission.csv', 'w') as f:
        f.writelines(results)
        

if __name__ == '__main__':

    test_data_transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    use_gpu = torch.cuda.is_available()

    batch_size = 32
    num_workers = 8

    test_file = '/Users/demerzel/PycharmProjects/cat-dog/data/test.txt'

    test_dataset = CatDogDataset(file_path=test_file, model='test', data_transforms=test_data_transforms)
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_datasize = len(test_dataset)

    num_classes = 2
    
    model = AlexNet(num_classes=2)
    model = torch.load('./output/epoch_70.pkl')

    if use_gpu:
        model = model.cuda()

    eval_model(model=model,use_gpu=use_gpu)