import os
from PIL import Image,ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms as T

class DAEDataset(Dataset):
    def __init__(self,dataset_path='./dataset/temp',img_type="img",resize=64,cropsize=64,
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.dataset_path = dataset_path
        self.img_type = img_type
        self.type_selector={"img":"Img","dimg":"Dimg"}
        self.imgs = self.load_dataset_folder()
        self.resize=resize
        self.cropsize=cropsize
        self.mean=mean
        self.std=std

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = Image.open(img).convert('L')
        temp=int((img.size[1]-img.size[0])/2)
        arround=0
        pad=(temp+arround,0+arround) if temp>=0 else (0+arround,arround-temp)
        transform_x=T.Compose([T.Pad(pad,fill=0), T.Resize((self.resize,self.resize), Image.ANTIALIAS),
                               T.ToTensor(),T.Normalize(mean=0.5,std=0.5)])
        img = transform_x(img)
        return img

    def __len__(self):
        return len(self.imgs)

    def load_dataset_folder(self):
        phase = self.type_selector[self.img_type]
        imgs = []
        img_dir = os.path.join(self.dataset_path, phase)

        img_fpath_list = sorted([os.path.join(img_dir, f)
                                for f in os.listdir(img_dir)
                                if f.endswith(('.jpg','.png'))])
        imgs.extend(img_fpath_list)
        return list(imgs)