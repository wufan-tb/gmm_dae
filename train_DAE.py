import torch,argparse,os
from torch.utils.data import DataLoader
from torchvision import utils
from model.AE_model import SI_DAE,DI_DAE
from dataset import DAEDataset

def train_DAE(args):
    ckpt_path=os.path.join(args.output_path,'checkpoint')
    sample_path=os.path.join(args.output_path,'train_samples')
    log_path=os.path.join(args.output_path,'train_log.log')
    with open(log_path,'a',encoding='utf-8') as f:
        f.writelines("\n\n====== New Training Loop ======\n\n")
        f.writelines("== Train argument: ==:\n")
        for item in args.__dict__.items():
            f.writelines(str(item)+"\n")
        f.writelines("\n"+"== Train log: ==")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    dataset = DAEDataset(dataset_path=args.dataset_path, img_type=args.img_type)
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)
    class_name = os.path.basename(args.dataset_path)
    model_select={"img":SI_DAE,"dimg":DI_DAE}
    model = model_select[args.img_type]().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.6)
    for epoch in range(args.train_epochs):
        model.train()
        if epoch==1:
            print("== first epoch done! ==")
        sample_size = 10
        for i, (img) in enumerate(loader): 
            img = img.to(args.device)
            out,fea,mse,ssim = model(img)
            loss=mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            if (epoch+1)%args.save_gap == 0 and i % (int(len(loader)/5)+1) == 0 :
                with open(log_path,'a',encoding='utf-8') as f:
                    f.writelines("epoch:{}-{}; loss:{:.3f}; lr:{:.5f}".format(epoch+1,i+1,loss.item(),lr)+'\n')

        if (epoch+1)%args.save_gap == 0:
            model.eval()
            sample = img[:sample_size]
            with torch.no_grad():
                out,_,_,_ = model(sample)
            utils.save_image(torch.cat([sample, out], 0),
                os.path.join(sample_path,"{}_{}_{}.jpg".format(class_name,args.img_type,epoch+1)),
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),)
            model.train()
            torch.save(model, os.path.join(ckpt_path,"{}_{}_{}.pt".format(class_name,args.img_type,epoch+1)))
            print('save samples and ckpt, at epoch {}'.format(epoch+1))
        scheduler.step()
    print("{} traing process done".format(args.img_type))
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_type', type=str, default='img', choices=['img','dimg'], help='input img or dimg')
    parser.add_argument('--batch_size', type=int, default='500')
    parser.add_argument('--train_epochs', type=int, default='100')
    parser.add_argument('--save_gap', type=int, default='10', help="save by this frequency")
    parser.add_argument('--dataset_path', type=str, default='dataset/ped2/train', help='dataset path')
    parser.add_argument('--output_path', type=str, default='output', help='path to save log and ckpt')
    parser.add_argument('--device', type=str, default='cuda', help='device number')
    parser.add_argument('--learning_rate', type=float, default='0.01', help='init img resize retio')
    args = parser.parse_args()
    args.img_type='img'
    train_DAE(args)
    args.img_type='dimg'
    train_DAE(args)
