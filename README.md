本库不包含数据集：
dataset:
    DIV2K_train_LR：http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
    DIV2K_train_HR：http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
    DIV2K_LSDIR_valid_LR：https://drive.google.com/file/d/1YUDrjUSMhhdx1s-O0I1qPa_HjW-S34Yj/view?usp=sharing
    DIV2K_LSDIR_valid_HR：https://drive.google.com/file/d/1z1UtfewPatuPVTeAAzeTjhEGk4dg2i8v/view?usp=sharing

环境配置：pip install -r requirements.txt
    

数据切片：
    python extract_subimages.py --input ../dataset/DIV2K_train_HR  --output ../dataset/DIV2K_train_patch_HR --crop_size=256 --step=128
    python extract_subimages.py --input ../dataset/DIV2K_train_LR  --output ../dataset/DIV2K_train_patch_LR --crop_size=64 --step=32

test_demo.py运行：
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir ./dataset --save_dir ./result/result_baseline --model_id 0
