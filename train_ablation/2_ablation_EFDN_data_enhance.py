import os
import logging
from collections import OrderedDict

from utils import utils_logger
import torch
import torch.nn as nn

from torch.utils.data import DataLoader,random_split,ConcatDataset
from torchvision import transforms
from torchvision.transforms import ToTensor

from make_datasets import MyGODataset, MyGODataset_valid,MyGODataset_eh
from models.MyGO_train import MyGO
from models.EFDN_train import EFDN



def main():
    logger_name='EFDN-ablution-conv1-3-data-enhance'
    logger_path=logger_name+'.log'
    utils_logger.logger_info(logger_name,
                             log_path=logger_path)
    logger = logging.getLogger(logger_name)

    epochs = 60
    Learning_rate = 1e-3
    save_path = "model_zoo/EFDN-best_model-ablution-conv1-3-data-enhance.pth"

    train_sets = r"/media/imcv100/31aa1c8f-3a4c-4c09-994b-93a53dde4192/NTIRE2025ESR/MyGO/datasets/DF2K_patch/train"
    train_sets_x4 = r"/media/imcv100/31aa1c8f-3a4c-4c09-994b-93a53dde4192/NTIRE2025ESR/MyGO/datasets/DF2K_patch/train_x4"

    valid_sets = r"/media/imcv100/31aa1c8f-3a4c-4c09-994b-93a53dde4192/NTIRE2025ESR/MyGO/datasets/DF2k_patch/valid"
    valid_sets_x4 = r"/media/imcv100/31aa1c8f-3a4c-4c09-994b-93a53dde4192/NTIRE2025ESR/MyGO/datasets/DF2k_patch/valid_x4"


    ToTensor = transforms.ToTensor()

    trainset = MyGODataset(train_sets, train_sets_x4, transform=ToTensor)

    trainset_flip=MyGODataset_eh(train_sets, train_sets_x4, flip=True,transform=ToTensor)
    trainset_rotate=MyGODataset_eh(train_sets, train_sets_x4, Rotate=True,transform=ToTensor)
    trainset_flip,_=random_split(trainset_flip,[0.1,0.9])
    trainset_rotate,_=random_split(trainset_rotate,[0.1,0.9])

    trainset=ConcatDataset([trainset,trainset_flip,trainset_rotate])

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)



    validset = MyGODataset(valid_sets, valid_sets_x4, transform=ToTensor)
    valid_loader = DataLoader(validset, batch_size=32, shuffle=True)

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EFDN()

    # 打开conv1x1与conv3x3串联模块
    for cell in model.cells:
        cell.conv2.with_13 = True
        cell.conv3.with_13 = True

        cell.conv2.with_idt = True
        cell.conv3.with_idt = True

    model = model.to(device)

    criterion = nn.L1Loss()
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    # record PSNR, runtime
    train_results = OrderedDict()
    train_results['loss'] = []
    train_results['psnr'] = []

    l_rate = []

    best_psnr = -float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for idx, (hr, lr) in enumerate(train_loader):
            hr = hr.to(device)
            lr = lr.to(device)
            sr = model(lr)

            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l_rate.append(scheduler.get_last_lr()[0])
            scheduler.step()
            train_results['loss'].append(loss.item())
            print(loss)
        avg_loss = sum(train_results['loss']) / len(train_results['loss'])
        print(avg_loss)
        logger.info('{:4d} --> Average loss{:.10f}'.format(epoch, avg_loss))

        model.eval()
        total_loss = 0.0
        total_fig = 0
        with torch.no_grad():
            for idx, (hr, lr) in enumerate(valid_loader):
                batch_size = hr.shape[0]
                hr = hr.to(device)
                lr = lr.to(device)
                sr = model(lr)
                loss = mse(sr, hr)

                total_fig += batch_size
                total_loss += loss * batch_size

            avg_loss = total_loss / total_fig
            current_psnr = 10 * torch.log10(1.0 ** 2 / (avg_loss + 1e-10))
            print(current_psnr)
            logger.info('{:4d} --> psnr{:.5f}'.format(epoch, current_psnr))

            if current_psnr > best_psnr:
                logger.info('new best model appears at epoch{:4d} --> psnr{:.5f}'.format(epoch, current_psnr))
                best_psnr = current_psnr
                best_model_state = model.state_dict().copy()  # 保存最佳模型参数
                torch.save(best_model_state, save_path)  # 保存到文件

    return


if __name__ == '__main__':
    main()