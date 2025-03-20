import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from fvcore.nn import FlopCountAnalysis
from utils.model_summary import get_model_activation, get_model_flops
from utils import utils_logger
from utils import utils_image as util


# 选择模型      模型ID等参数 / 训练设备
def select_model(args, device):
    # 模型ID根据提交顺序依次分配。
    # 不同网络采用[0,1]或[0,255]的输入范围进行训练，具体范围由人工手动确定。
    model_id = args.model_id
    if model_id == 0:
        # 基线：NTIRE 2023 高效超分辨率挑战赛总体性能第1名
        # 边缘增强特征蒸馏网络：高效超分辨率方法
        # arXiv预印本：https://arxiv.org/pdf/2204.08759
        # 原始代码：https://github.com/icandle/EFDN
        # 检查点：EFDN_gv.pth
        name = f"{model_id:02}_EFDN_baseline"                       # 模型名称
        data_range = 1.0                                            # 数据范围
        from models.team00_EFDN import EFDN                         # 导入EFDN模型
        model = EFDN()                                              # 实例化一个空的 EFDN 模型
        model_path = os.path.join('model_zoo', 'team00_EFDN.pth')   # 从当前路径 查找 模型(检查点)路径
        model.load_state_dict(torch.load(model_path), strict=True)  # 从检查点加载模型参数
    elif model_id == 1:
        pass # ---- Put your model here as below ---                # 给用户留下的接口
        # from models.team01_[your_model_name] import [your_model_name]
        # name, data_range = f"{model_id:02}_[your_model_name]", [255.0 / 1.0] # You can choose either 1.0 or 255.0 based on your own model
        # model_path = os.path.join('model_zoo', 'team01_[your_model_name].pth')
        # model = [your_model_name]()
        # model.load_state_dict(torch.load(model_path), strict=True)
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    # print(model)
    model.eval()                            # 切换为评估模式
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)                # 放到GPU上测试
    return model, name, data_range, tile


# 选择数据集    数据所在文件夹 / 测试or验证模式
def select_dataset(data_dir, mode):
    if mode == "test":  # 测试数据集 DIV2K_LSDIR_test
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "x4.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "DIV2K_LSDIR_test_HR/*.png")))
        ]

    elif mode == "valid":   # 验证数据集 DIV2K_LSDIR_valid set
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "x4.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "DIV2K_LSDIR_valid_HR/*.png")))
        ]
    else:
        raise NotImplementedError(f"{mode} is not implemented in select_dataset")
    
    return path


# 前向传播      图像 / 模型 / 切块宽度 / 重合宽度 / 超分倍数
def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        # 整张图片进行推理
        output = model(img_lq)
    else:
        # 将图片切分成小块推理
        b, c, h, w = img_lq.size()      # 张量记为：batch/channel/higw/wight
        tile = min(tile, h, w)          # 确保块大小不超过图像尺寸
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap                            # 计算：块间步幅 = 切块宽度 - 重合宽度
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]  # 生成分块高度索引，由于列表不包含右边界，所以在列表中添加 [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]  # 生成分块宽度索引
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)       # 存储  各块预测结果的累加能量（对应高分辨率图像的像素值）
        W = torch.zeros_like(E)                                 # 存储  各块有效区域的权重（用于后续归一化）。

        for h_idx in h_idx_list:                                # 遍历所有图像块
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]  # 通道全保留，行列切块
                out_patch = model(in_patch)                     # 用切块后的图像推理
                out_patch_mask = torch.ones_like(out_patch)     # 生成Size相同的全1张量，掩码用于求均值

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)          # 将不同图像块的超分结果叠加
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)     # 根据掩码求均值，生成无重叠超分图像
        output = E.div_(W)

    return output


# 运行          模型 / 模型名称 / 1 / 不切块 / 记录 / 设备 / 接口 / 模式
def run(model, model_name, data_range, tile, logger, device, args, mode="test"):

    sf = 4                              # 超分倍数
    border = sf                         
    results = dict()                    # 创建空字典以保存所有结果
    results[f"{mode}_runtime"] = []     # 运行时间
    results[f"{mode}_psnr"] = []        # 峰值信噪比
    if args.ssim:
        results[f"{mode}_ssim"] = []    # 结构相似性指数
    # results[f"{mode}_psnr_y"] = []
    # results[f"{mode}_ssim_y"] = []

    # --------------------------------
    # 设置路径
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode)             # 数据集路径
    save_path = os.path.join(args.save_dir, model_name, mode)   # 保存路径
    util.mkdir(save_path)                                       # 确保“保存路径”存在

    start = torch.cuda.Event(enable_timing=True)                # 计算 GPU 运行时间
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_lr, img_hr) in enumerate(data_path):

        # --------------------------------
        # (1) 加载低分辨率图像 img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))      # 提取图像名称与扩展名
        img_lr = util.imread_uint(img_lr, n_channels=3)                 # 读取低分辨率图像
        img_lr = util.uint2tensor4(img_lr, data_range)                  # 图像归一化并转张量
        img_lr = img_lr.to(device)                                      # 加载到 GPU

        # --------------------------------
        # (2) 图像推理 img_sr
        # --------------------------------
        start.record()
        img_sr = forward(img_lr, model, tile)                           # 前向传播
        end.record()
        torch.cuda.synchronize()                                        # 强制同步所有尚未完成的 CUDA 操作
        results[f"{mode}_runtime"].append(start.elapsed_time(end))      # 计算运行时间 ms
        img_sr = util.tensor2uint(img_sr, data_range)                   # 将张量转为图像

        # --------------------------------
        # (3) 加载高分辨率图像 img_hr
        # --------------------------------
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, sf)

        # --------------------------------
        # 计算 PSNR 和 SSIM
        # --------------------------------

        # print(img_sr.shape, img_hr.shape)
        psnr = util.calculate_psnr(img_sr, img_hr, border=border)       # 根据 推理图像 与 高清图像 计算 PSNR
        results[f"{mode}_psnr"].append(psnr)

        if args.ssim:                                                   # 计算 SSIM
            ssim = util.calculate_ssim(img_sr, img_hr, border=border)
            results[f"{mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        # if np.ndim(img_hr) == 3:  # RGB image
        #     img_sr_y = util.rgb2ycbcr(img_sr, only_y=True)
        #     img_hr_y = util.rgb2ycbcr(img_hr, only_y=True)
        #     psnr_y = util.calculate_psnr(img_sr_y, img_hr_y, border=border)
        #     ssim_y = util.calculate_ssim(img_sr_y, img_hr_y, border=border)
        #     results[f"{mode}_psnr_y"].append(psnr_y)
        #     results[f"{mode}_ssim_y"].append(ssim_y)
        # print(os.path.join(save_path, img_name+ext))
            
        # --- Save Restored Images ---
        # util.imsave(img_sr, os.path.join(save_path, img_name+ext))

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"])      # 平均运行时间
    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])               # 平均PSNR
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])           # 平均 SSIM
    # results[f"{mode}_ave_psnr_y"] = sum(results[f"{mode}_psnr_y"]) / len(results[f"{mode}_psnr_y"])
    # results[f"{mode}_ave_ssim_y"] = sum(results[f"{mode}_ssim_y"]) / len(results[f"{mode}_ssim_y"])
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memory", results[f"{mode}_memory"]))  # Memery
    logger.info("------> Average runtime of ({}) is : {:.6f} milliseconds".format("test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))
    logger.info("------> Average PSNR of ({}) is : {:.6f} dB".format("test" if mode == "test" else "valid", results[f"{mode}_ave_psnr"]))

    return results


# 主函数        接口参数
# 用于评估图像超分辨率模型性能
def main(args):
    # 使用 log 文件记录实验过程
    utils_logger.logger_info("NTIRE2025-EfficientSR", log_path="NTIRE2025-EfficientSR.log")
    logger = logging.getLogger("NTIRE2025-EfficientSR")

    # --------------------------------
    # 基础设置
    # --------------------------------
    torch.cuda.current_device()                 # 获取当前默认GPU设备索引（通常为0）
    torch.cuda.empty_cache()                    # 释放未被引用的GPU缓存内存[4,5,6](@ref)
    torch.backends.cudnn.benchmark = False      # 关闭CuDNN自动优化算法功能[7,8,9](@ref)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 定义设备变量

    json_dir = os.path.join(os.getcwd(), "results.json")    # json 文件的存储路径
    if not os.path.exists(json_dir):                        # 不存在就新建
        results = dict()
    else:
        with open(json_dir, "r") as f:                      # 存在的话就加载之前的结果（用于迭代2）
            results = json.load(f)

    # --------------------------------
    # 加载模型
    # --------------------------------
    model, model_name, data_range, tile = select_model(args, device)    # 加载模型
    logger.info(model_name)                                             # 记录模型名称

    # if model not in results:
    if True:
        # --------------------------------
        # 恢复镜像
        # --------------------------------

        # 使用验证集 DIV2K_LSDIR_valid 进行验证
        valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")
        # 记录 峰值信噪比 PSNR & 运行时间 runtime
        results[model_name] = valid_results

        # 主办方对测试集 DIV2K_LSDIR_test 的推理
        if args.include_test:
            test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
            results[model_name].update(test_results)

        input_dim = (3, 256, 256)  # 设置输入维度
        activations, num_conv = get_model_activation(model, input_dim)              # 统计显存占用（M）
        activations = activations/10**6                                         
        logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        # 上一届 NTIRE_ESR 挑战赛中的 FLOPs 计算
        # flops = get_model_flops(model, input_dim, False)
        # flops = flops/10**9
        # logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        # fvcore 在 NTIRE2025_ESR 中用于 FLOPs 计算
        input_fake = torch.rand(1, 3, 256, 256).to(device)                          # 统计 FLOPS （G）
        flops = FlopCountAnalysis(model, input_fake).total()
        flops = flops/10**9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))          # 统计参数量（M）
        num_parameters = num_parameters/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
        results[model_name].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})

        with open(json_dir, "w") as f:
            json.dump(results, f)
    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        val_psnr = f"{v['valid_ave_psnr']:2.2f}"
        val_time = f"{v['valid_ave_runtime']:3.2f}"
        mem = f"{v['valid_memory']:2.2f}"
        
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.include_test:
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, acts, mem, conv)
    with open(os.path.join(os.getcwd(), 'results.txt'), "w") as f:
        f.write(s)


# 对外接口
if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2025-EfficientSR")
    parser.add_argument("--data_dir", default="../", type=str)                  # 数据集的路径
    parser.add_argument("--save_dir", default="../results", type=str)           # 保存结果的路径
    parser.add_argument("--model_id", default=0, type=int)                      # 模型ID，用于选择模型
    parser.add_argument("--include_test", action="store_true", help="Inference on the `DIV2K_LSDIR_test` set")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")   # 计算结构相似性指数

    args = parser.parse_args()      # 实例化接口参数
    pprint(args)                    # 输出接口参数
    main(args)                      # 运行主函数
