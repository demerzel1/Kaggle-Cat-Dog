from torch.utils.data import Dataset
from PIL import Image

def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print('Can not read image: {}'.format(path))

class CatDogDataset(Dataset):
    def __init__(self, file_path, dataset='', model='train', data_transforms=None, loader= default_loader):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        self.img_path = []
        self.model = model
        for line in lines:
            self.img_path.append(line)
        if self.model == 'train'
            class_to_idx = {'dog': 0, 'cat': 1}
            slef.label = [class_to_idx[os.path.basename(line)[1].split('.')[0]] for line in lines]