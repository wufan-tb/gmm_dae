import torch,argparse,os,pickle
from torch.utils.data import DataLoader
from torchvision import utils
from dataset import DAEDataset
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import mixture 

def get_feature(args):
    model = torch.load(args.ckpt,map_location="cuda:0")
    model = model.to(args.device)
    model.eval()
    
    dataset = DAEDataset(dataset_path=args.dataset_path, img_type=args.img_type)
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)
    class_name = os.path.basename(args.dataset_path)
    Feature=None
    for i, (img) in enumerate(loader): 
        img = img.to(args.device)
        out,fea,mse,ssim = model(img)
        if i==0:
            Feature=fea.cpu().detach().numpy()
        else:
            Feature=np.append(Feature,fea.cpu().detach().numpy(),axis=0)
    return np.reshape(Feature,(Feature.shape[0],-1))

def train_gmm(args):
    print('==={} gmm start training==='.format(args.img_type))
    for item in args.__dict__.items():
        print(item)
    with torch.no_grad():
        X=get_feature(args)
    gmm = mixture.GaussianMixture(n_components=args.gmm_components,verbose=1)
    gmm.fit(X)
    theta={}
    theta['pi']=gmm.weights_
    theta['miu']=gmm.means_
    theta['sigma']=gmm.covariances_
    with open(os.path.join(args.output_path,"{}_GmmTheta.pkl".format(args.img_type)), "wb") as f:   
        pickle.dump(theta, f)
    print('==={} gmm training done==='.format(args.img_type))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--img_type', type=str, default='img', choices=['img','dimg'], help='input img or dimg')
    parser.add_argument('--batch_size', type=int, default='1000')
    parser.add_argument('--dataset_path', type=str, default='dataset/ped2/train', help='dataset path')
    parser.add_argument('--output_path', type=str, default='output/pkl', help='path to save log and ckpt')
    parser.add_argument('--device', type=str, default='cuda', help='device number')   
    parser.add_argument('--gmm_components', type=int, default='15') 
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    args.img_type='img'
    args.ckpt="output/checkpoint/train_img_100.pt"
    train_gmm(args)
    
    args.img_type='dimg'
    args.ckpt="output/checkpoint/train_dimg_100.pt"
    train_gmm(args)