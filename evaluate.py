import argparse,sys,os,torch,cv2,pickle,time
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import transforms as T
from torchvision import utils
from sklearn.metrics import roc_auc_score
from scipy.stats import multivariate_normal

sys.path.append('pytorch_yolov5/')
from models.experimental import *
from utils.datasets import *
from utils.utils import *

def ae_preprocess(img_cv2):
    img_PIL=Image.fromarray(cv2.cvtColor(img_cv2,cv2.COLOR_BGR2RGB))
    img=img_PIL.convert('L')
    temp=int((img.size[1]-img.size[0])/2)
    arround=0
    pad=(temp+arround,0+arround) if temp>=0 else (0+arround,arround-temp)
    transform=T.Compose([T.Pad(pad,fill=0), T.Resize((64,64), Image.ANTIALIAS),
                            T.ToTensor(),T.Normalize(mean=0.5,std=0.5)])
    img_PIL=transform(img)
    img_tensor=img_PIL[np.newaxis,:]
    return img_tensor.to('cuda')

def calcu_proba(theta,test_x):
    k=theta["pi"].shape[0]
    q = np.zeros((k, 1))
    for i in range(k):
        q[i, :] = theta['pi'][i]*multivariate_normal.pdf(test_x,mean=theta['miu'][i],cov=theta['sigma'][i, ...])
    return np.log(1e-5+q.sum())

def detect(save_img=False):    
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    sample_path="output/test_samples"
    os.makedirs(sample_path,exist_ok=True)
    device = torch_utils.select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    simg_dae_model = torch.load(opt.SIDAE_ckpt,map_location="cuda")
    simg_dae_model = simg_dae_model.to('cuda')
    simg_dae_model.eval()
    dimg_dae_model = torch.load(opt.DIDAE_ckpt,map_location="cuda")
    dimg_dae_model = dimg_dae_model.to('cuda')
    dimg_dae_model.eval()
    with open(r"output/pkl/img_GmmTheta.pkl", "rb") as f:    
        img_theta=pickle.load(f)
    with open(r"output/pkl/dimg_GmmTheta.pkl", "rb") as f:    
        dimg_theta=pickle.load(f)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    MSE_FEA_List=[]
    SI_feature=[]
    DI_feature=[]
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            Dimg=cv2.imread(os.path.join(os.path.dirname(opt.source),"Dimg",Path(p).name))
            total_ae_time=0.0
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                patch_index=0
                total_ae_time=time.time()
                psnr1=[]
                psnr2=[]
                likelihood1=[]
                likelihood2=[]
                for *xyxy, conf, cls in det:
                    patch_index+=1
                    label = '%s %.2f' % (names[int(cls)], conf)
                    img_name=Path(p).name[0:-4]+"_patch{}.jpg".format(patch_index)
                    xmin,ymin,xmax,ymax = (int(xyxy[0]), int(xyxy[1]),int(xyxy[2]), int(xyxy[3]))
                    if (xmax-xmin)*(ymax-ymin)> opt.min_boxsize:
                        img_in=ae_preprocess(im0[ymin:ymax,xmin:xmax,:])
                        dimg_in=ae_preprocess(Dimg[ymin:ymax,xmin:xmax,:])
                        recon1,fea1,mse1,_=simg_dae_model(img_in)
                        recon2,fea2,mse2,_=dimg_dae_model(dimg_in)
                        psnr1.append(10*np.log10(255/mse1.item()))
                        psnr2.append(10*np.log10(255/mse2.item()))
                        likelihood1.append(calcu_proba(img_theta,fea1.flatten().cpu()))
                        likelihood2.append(calcu_proba(dimg_theta,fea2.flatten().cpu()))
                        SI_feature.append(np.array(fea1.flatten().cpu()))
                        DI_feature.append(np.array(fea1.flatten().cpu()))
                        if names[int(cls)] != "person" or conf<=0.4:
                            utils.save_image(torch.cat([img_in, recon1,dimg_in,recon2], 0),
                            os.path.join(sample_path,"{}|{:.1f}|{:.1f}|{:.1f}|{:.1f}.jpg".format(Path(p).name[0:-4],psnr1[-1],psnr2[-1],likelihood1[-1],likelihood2[-1])),
                            nrow=1,
                            normalize=True,
                            range=(-1, 1),)
                MSE_FEA_List.append([psnr1,psnr2,likelihood1,likelihood2])
                total_ae_time=time.time()-total_ae_time
            else:
                MSE_FEA_List.append([[],None,None,None])
            # Print time (inference + NMS)
            print('%sDone.(detect: %.3fs),(total dae: %.3fs).' % (s,t2-t1,total_ae_time))

    print('Done. (%.3fs)' % (time.time() - t0))
    return MSE_FEA_List,SI_feature,DI_feature

def moving_avg(X,belta=0.25):
    Y=X.copy()
    for i in range(X.shape[0]):
        if i >= 1:
            Y[i]=belta*Y[i]+(1-belta)*Y[i-1]
        else:
            pass
    return Y

def calculate_AUROC(MSE_FEA_List,score_lambda,ground_th,use_slide_avg=False):
    prediction=[]
    for (psnr1,psnr2,likelihood1,likelihood2) in MSE_FEA_List:
        max_score=-50
        for i in range(len(psnr1)):
            score=100-np.dot(np.array([psnr1[i],psnr2[i],likelihood1[i],likelihood2[i]]),score_lambda)
            if score>max_score:
                max_score=score
        prediction.append(max_score)
    prediction=np.array(prediction)
    if use_slide_avg:
        prediction=moving_avg(prediction)
    prediction-=np.min(prediction)
    prediction/=max(1e-5,np.max(prediction))
    
    return roc_auc_score(ground_th,prediction), prediction

def grid_search(MSE_FEA_List,ground_th,search_step_size=0.1):
    max_lambda=None
    max_roc=-50
    prediction=None
    for i in range(int(1/search_step_size)+1):
        lambda1=search_step_size*i
        for j in range(int((1-lambda1)/search_step_size)+1):
            lambda2=search_step_size*j
            for k in range(int((1-lambda1-lambda2)/search_step_size)+1):
                lambda3=search_step_size*k
                lambda4=1-lambda1-lambda2-lambda3
                temp=np.array([lambda1,lambda2,lambda3,lambda4])
                roc, prediction=calculate_AUROC(MSE_FEA_List,temp,ground_th)
                if max_roc<roc:
                    max_roc=roc
                    max_lambda=temp
    return max_roc,max_lambda,prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='pytorch_yolov5/weights/yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--SIDAE_ckpt', type=str, default='output/checkpoint/train_img_100.pt')
    parser.add_argument('--DIDAE_ckpt', type=str, default='output/checkpoint/train_dimg_100.pt')
    parser.add_argument('--source', type=str, default='dataset/ped2/test/Img', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output/pkl', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--min_boxsize', type=int, default=10, help='box area less than it will be ignored')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    os.makedirs(opt.output, exist_ok=True)
    with torch.no_grad():
        MSE_FEA_List,SI_fea,DI_fea = detect()
    with open(os.path.join(opt.output,"MSE_FEA_List.pkl"), "wb") as f:   
        pickle.dump(MSE_FEA_List, f)
    with open(os.path.join(opt.output,"SI_FEATURE.pkl"), "wb") as f:   
        pickle.dump(SI_fea, f)
    with open(os.path.join(opt.output,"DI_FEATURE.pkl"), "wb") as f:   
        pickle.dump(DI_fea, f)
    arr=np.load('dataset/frame_labels_ped2.npy')
    ground_th=arr[0]

    print("SI+DAE AUROC: ",calculate_AUROC(MSE_FEA_List,np.array([1,0,0,0]),ground_th)[0])
    print("DI+DAE AUROC: ",calculate_AUROC(MSE_FEA_List,np.array([0,1,0,0]),ground_th)[0])
    print("SI+GMM AUROC: ",calculate_AUROC(MSE_FEA_List,np.array([0,0,1,0]),ground_th)[0])
    print("DI+GMM AUROC: ",calculate_AUROC(MSE_FEA_List,np.array([0,0,0,1]),ground_th)[0])
    max_roc,best_lambda,prediction=grid_search(MSE_FEA_List,ground_th)
    print("max AUROC:",max_roc,"; best lambda:",'[ %.3f , %.3f , %.3f , %.3f ]'%(best_lambda[0],best_lambda[1],best_lambda[2],best_lambda[3]))
    print("max AUROC after filter: ",calculate_AUROC(MSE_FEA_List,best_lambda,ground_th,use_slide_avg=True)[0])
