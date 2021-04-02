# %%
import subprocess,os,natsort,shutil,argparse

def dataset_prepare(args):
    dataset_path=os.path.join(args.dataset_name,"frames_"+args.train_or_test)
    output_path=os.path.join(args.dataset_name,args.train_or_test)
    # * 把所有frames下的文件夹合成对应的dimg文件夹
    file_list=natsort.natsorted(os.listdir(dataset_path))
    for item in file_list:
        if 'MOIF' in item:
            pass
        else:
            fpath=os.path.join(dataset_path,item)
            sys_cmd="python motion_info.py --save -j {} -s {} -l {}".format(args.motion_info_type,fpath,args.t_frames)
            child = subprocess.Popen(sys_cmd,shell=True)
            child.wait()
            print(item,"done, Return code:",child.returncode)
            
    file_list=natsort.natsorted(os.listdir(dataset_path))
    for item in file_list:
        if 'MOIF' in item:
            fpath=os.path.join(dataset_path,item)
            for i,imgname in enumerate(natsort.natsorted(os.listdir(fpath))):
                os.rename(os.path.join(fpath,imgname),os.path.join(fpath,item[0:-5]+"_{}.jpg".format(i+1)))
                if i==args.cover_frames_number:
                    for j in range(args.cover_frames_number):
                        shutil.copy(os.path.join(fpath,item[0:-5]+"_{}.jpg".format(i+1)),os.path.join(fpath,item[0:-5]+"_{}.jpg".format(j+1)))
        else:
            fpath=os.path.join(dataset_path,item)
            for i,imgname in enumerate(natsort.natsorted(os.listdir(fpath))):
                os.rename(os.path.join(fpath,imgname),os.path.join(fpath,item+"_{}.jpg".format(i+1)))
            
    print("dimg generated!")

    # * 根据yolov5的识别结果生成小patch,仅针对训练集
    if args.train_or_test=="train":
        file_list=natsort.natsorted(os.listdir(dataset_path))
        for item in file_list:
            if 'MOIF' in item:
                pass
            else:
                fpath=os.path.join(dataset_path,item)
                sys_cmd="python yolov5_generate_patch.py --source {} --output {} --img-size {} --conf-thres {} --iou-thres {} --min_boxsize {}".format(
                    fpath,output_path,args.detector_imgsize,args.detector_conf,args.detector_nms,args.detector_min_boxsize)
                child = subprocess.Popen(sys_cmd,shell=True)
                child.wait()
                print(item,"done, Return code:",child.returncode)
        print("patch img and dimg generated!")
    else:
        Img_path=os.path.join(output_path,"Img")
        Dimg_path=os.path.join(output_path,"Dimg")
        os.makedirs(Img_path,exist_ok=True)
        os.makedirs(Dimg_path,exist_ok=True)
        file_list=natsort.natsorted(os.listdir(dataset_path))
        for item in file_list:
            if 'MOIF' in item:
                fpath=os.path.join(dataset_path,item)
                for i,imgname in enumerate(natsort.natsorted(os.listdir(fpath))):
                    shutil.move(os.path.join(fpath,imgname),os.path.join(Dimg_path,imgname))
            else:
                fpath=os.path.join(dataset_path,item)
                for i,imgname in enumerate(natsort.natsorted(os.listdir(fpath))):
                    shutil.move(os.path.join(fpath,imgname),os.path.join(Img_path,imgname))
        print("move img and dimg done!")
        
    print("{} data preparation Done!".format(args.train_or_test))
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_info_type', type=int, default='1', help='3:diff frame, 1：dynamic image')
    parser.add_argument('--t_frames', type=int, default='10', help='number of multi-pre-frames, include local one')
    parser.add_argument('--dataset_name', type=str, default='dataset/ped2', help='dataset path')
    parser.add_argument('--train_or_test', type=str, default='test', choices=['train','test'])
    parser.add_argument('--cover_frames_number', type=int, default='10',help='recommand equal to t_frames')
    
    parser.add_argument('--detector_imgsize', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--detector_conf', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--detector_nms', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--detector_min_boxsize', type=int, default=10, help='box area less than it will be ignored')
    args = parser.parse_args()
    
    dataset_prepare(args)
    args.train_or_test="train"
    dataset_prepare(args)