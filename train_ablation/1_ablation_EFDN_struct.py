import sys
import os
import logging
from collections import OrderedDict

# r 用于忽略转义字符，用\不能加r，用/随便
root = '/home/u202410081000082/project/NTIRE2025_ESR'
sys.path.append(root)

from utils import utils_logger
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

from models.make_datasets import MyGODataset,MyGODataset_valid
from models.a1_EFDN_train import EFDN
# from parent_package.models.MyGO_train import MyGO


def main():
    epochs = 30             # 30 epoch
    batch_size = 64         # 每个 batch 64*8 张图
    Learning_rate = 1e-3    # 学习率 0.001*3      缩放 batchsize 和 学习率:小倍数线性,大倍数开方
    save_path = root + '/model_zoo/3_best_model_struct_all-depth2_bs64_lr1e-3.pth'

    criterion = nn.L1Loss()                                             # L1 损失函数
    mse = nn.MSELoss()                                                  # 均方误差损失函数
    
    if not os.path.exists(root + '/result/struct_all-depth2_bs64_lr1e-3'):
    	os.makedirs(root + '/result/struct_all-depth2_bs64_lr1e-3')
    	print(f"文件夹 创建成功")
    else:
    	print(f"文件夹  已存在")

    logger_name = root + '/result/struct_all-depth2_bs64_lr1e-3/train'
    logger_path = logger_name + '.log'
    utils_logger.logger_info(logger_name,log_path=logger_path)
    logger = logging.getLogger(logger_name)

    train_sets = root + '/dataset/DIV2K_train_patch_HR'                # 训练 HR
    train_sets_x4 = root + '/dataset/DIV2K_train_patch_LR'             # 训练 LR
    valid_sets = root + '/dataset/DIV2K_LSDIR_valid_HR'                # 测试 HR
    valid_sets_x4 = root + '/dataset/DIV2K_LSDIR_valid_LR'             # 测试 LR

    ToTensor = transforms.ToTensor()
    trainset = MyGODataset(train_sets, train_sets_x4, transform=ToTensor)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # validset = MyGODataset(valid_sets, valid_sets_x4, transform=ToTensor)
    validset = MyGODataset_valid(valid_sets, valid_sets_x4, transform=ToTensor)     # 修改
    valid_loader = DataLoader(validset, batch_size=1, shuffle=True)                 # 测试 batch_size = 1

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EFDN()
    # EDBB 消融实验
    for cell in model.cells:
        cell.conv2.with_13 = True   # 1*1-3*3分支
        cell.conv3.with_13 = True
        cell.conv2.depth_multiplier = 2   # 1*1-3*3 多通道
        cell.conv3.depth_multiplier = 2
        cell.conv2.with_idt = True  # 残差分支
        cell.conv3.with_idt = True


    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)  # Adam 优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))   # 余弦退火学习率调度

    # 计算参数量
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    # 用于记录 PSNR, runtime
    train_results = OrderedDict()
    train_results['loss'] = []
    train_results['psnr'] = []

    l_rate = []                     # 记录学习率的变化
    best_psnr = -float('inf')       # 记录最佳的 PSNR
    best_model_state = None         # 记录最佳模型状态

    for epoch in range(epochs):
        model.train()
        for idx, (hr, lr) in enumerate(train_loader):
            hr = hr.to(device)
            lr = lr.to(device)
            sr = model(lr)          # 前向传播进行超分

            loss = criterion(sr, hr)    # L1 损失
            optimizer.zero_grad()   # 清除优化器梯度
            loss.backward()         # 反向传播
            optimizer.step()        # 更新参数

            l_rate.append(scheduler.get_last_lr()[0])
            scheduler.step()        # 更新学习率
            train_results['loss'].append(loss.item())
            print(f"Epoch [{epoch + 1}/{epochs}], 损失值: {loss.item():.4f}")
        avg_loss = sum(train_results['loss']) / len(train_results['loss'])
        print(f"平均损失值: {avg_loss:.4f}")
        logger.info('{:4d} --> Average loss{:.10f}'.format(epoch, avg_loss))

        model.eval()
        # total_loss = 0.0
        total_PSNR = 0.0
        total_fig = 0
        with torch.no_grad():
            for idx, (hr, lr) in enumerate(valid_loader):
                batch_size = hr.shape[0]
                hr = hr.to(device)
                lr = lr.to(device)
                sr = model(lr)
                loss = mse(sr, hr)

                total_fig += batch_size             # 总图片数量
                # total_loss += loss * batch_size     # 总损失
                total_PSNR += batch_size * 10 * torch.log10(1.0 ** 2 / (loss + 1e-10))

            # avg_loss = total_loss / total_fig
            # current_psnr = 10 * torch.log10(1.0 ** 2 / (avg_loss + 1e-10))
            current_psnr = total_PSNR / total_fig            

            print(f"PSNR: {current_psnr:.4f}")
            logger.info('{:4d} --> psnr{:.5f}'.format(epoch, current_psnr))

            if current_psnr > best_psnr:
                logger.info('new best model appears at epoch{:4d} --> psnr{:.5f}'.format(epoch, current_psnr))
                best_psnr = current_psnr
                best_model_state = model.state_dict().copy()  # 保存最佳模型参数
                torch.save(best_model_state, save_path)  # 保存到文件

    return


if __name__ == '__main__':
    main()
