## GMM-DAE



A Pytorch Re-Implement Trial (Not Official) of paper: [Video Anomaly Detection by Estimating Likelihood
of Representations](https://arxiv.org/pdf/2012.01468.pdf)



**Unfortunately, I haven't got the accuracy as good as the paper mentioned.**（maybe I have missed some details, any help will be appreciated）



**Here are something you may think it's valuable.**

- a better object-detector result can get better AUROC,(for example, large image size, regard smaller object box, use different conf-thres(train > test))
- I get same result as the author in Ped2, yet big margin in Avenue and ShanghaiTech (nearly less ten point)
- Score Calculation and Score Smoothing can get lead to a great improvement(you can find more details in [this repo](https://github.com/fjchange/object_centric_VAD) )

### Requirements

> **pytorch** >=1.5.0 ( I use 1.5.0 )
>
> **scikit-learn**

### Framework Overview

The framework include Three Parts:

1. dataset prepare ,contain object detector(which I use [yolov5](https://github.com/ultralytics/yolov5)) and computer dynamic image;
2. train denoised auto-encoder;
3. get feature cluster center(train GMM);
4. caculate anomaly score(evaluate);

### Datasets

You can get the download link from [here](github.com/StevenLiuWen/ano_pred_cvpr2018)

### Training:

1. prepare data

```python
python prepare_dataset.py
```

2. train DAE

```python
python train_DAE.py
```

3. train GMM

```python
python train_GMM.py
```

**details about parameters seen in scripts**

### Testing:

1. ```python
   python evaluate.py 
   ```

   **more visualization tools can find in notebook: test.ipynb**

### To Do List:

- [ ] try knowledge distillation
- [ ] try cluster loss



If you find this useful, please cite works as follows:

```
 misc{object_centrci_VAD,
     author = {Wu Fan},
     title = { A Implementation of {GMM-DAE} Using {Pytorch}},
     year = {2020},
     howpublished = {\url{https://github.com/wufan-tb/gmm_dae}}
  }
```

