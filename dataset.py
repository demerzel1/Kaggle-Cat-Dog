from torch.utils.data import Dataset
from PIL import Image
import os

def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print('Can not read image: {}'.format(path))

class CatDogDataset(Dataset):
    def __init__(self, file_path, model='train', data_transforms=None, loader=default_loader):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        self.img_path = []
        self.model = model
        for line in lines:
            self.img_path.append(line)
        if self.model == 'train':
            class_to_idx = {'dog': 0, 'cat': 1}
            # for line in lines:
            #     print(line)
            #     print(os.path.basename(line).split('.')[0])
            #     print(class_to_idx[os.path.basename(line).split('.')[0]])
            self.label = [class_to_idx[os.path.basename(line).split('.')[0]] for line in lines]
        self.data_transforms = data_transforms
        self.loader = loader

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img_path = self.img_path[item]
        img = self.loader(img_path)
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print('Can not transform image: {}'.format(img_path))
        if self.model == 'train':
            label = self.label[item]
            return img, label
        else:
            return img

            
            
