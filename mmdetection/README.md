# Training

## HTC_ResNext101_64x4d_FPN_TTA

```
    python tools\train.py mmdetection\configs\boostcamp\htc_Res\HTC_submission.py
```

## Cascade_SwinT_PAFPN

```
    python tools\train.py mmdetection\configs\boostcamp\Cascade_SwinT_PAFPN\Cascade_SwinT_PAFPN.py
```
## Cascade_SwinT_FPN

```
    python tools\train.py \mmdetection\configs\boostcamp\Cascade_SwinT_PAFPN\Cascade_SwinT_FPN.py
```

# Inference
```
    python tools\test.py <config file 주소> <pth 주소> --out <pkl 원하는 주소>
```
# To csv
```
    python pkl_to_submission.py --pkl <pkl_file> --csv <output 주소>
```

# Utils

## Bbox visualization

        Visualization code in visualization directory
    
    - vis_GT.ipynb: visualization of training dataset
    - vis_test.ipynb: visualization of inference result of test dataset
    - vis_train.ipynb: comparison of train dataset inference result and GT

## Stratified K-Fold

        Straitified K-Fold split implemented in strat directory
    
    - strat_kfold_mmdet.ipynb: json file made for mmdetection library
    - strat_kfold.ipynb: json file made for boostcamp assignment



