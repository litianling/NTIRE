# -*- coding: utf-8 -*-
# 官方版
import torch
import torch.nn as nn
import torch.nn.functional as F


# ESA模块（通道数,卷积）    维度不变    实现注意力机制、特征加权
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)                    # 1*1 cnn           n->f
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)     # 3*3 cnn   缩小
        self.conv_max = conv(f, f, kernel_size=3, padding=1)            # 3*3 cnn   不变
        self.conv3 = conv(f, f, kernel_size=3, padding=1)               # 3*3 cnn   不变
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)              # 3*3 cnn   不变
        self.conv_f = conv(f, f, kernel_size=1)                         # 1*1 cnn   不变
        self.conv4 = conv(f, n_feats, kernel_size=1)                    # 1*1 cnn           n->f
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        c1_ = (self.conv1(x)) 
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        return x * m


# 1×1卷积（通道数）         维度不变    对通道信息进行重新加权和特征融合
class conv(nn.Module):
    def __init__(self, n_feats):
        super(conv, self).__init__()
        self.conv1x1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0) 
        self.act = nn.PReLU(num_parameters=n_feats)
    def forward(self, x):
        return self.act(self.conv1x1(x))


# EFDB 单元（48通道,启用动态计算,禁用重参数化,默认层数,禁用13*13卷积核）    维度不变    特征提取
class Cell(nn.Module):
    def __init__(self, n_feats=48, dynamic = True, deploy = False, L= None, with_13=False):
        super(Cell, self).__init__()
        
        self.conv1 = conv(n_feats)                              # 维度不变的1*1卷积
        self.conv2 = EDBB_deploy(n_feats,n_feats)               # 两个EDBB 模块
        self.conv3 = EDBB_deploy(n_feats,n_feats)

        self.branch = nn.ModuleList([nn.Conv2d(n_feats, n_feats//2, 1, 1, 0) for _ in range(4)])    # 四个通道数减半的 1*1 卷积
        self.fuse = nn.Conv2d(n_feats*2, n_feats, 1, 1, 0)      # 通道数减半的 1*1 卷积
        self.att = ESA(n_feats, nn.Conv2d)                      # ESA 模块

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        # 四个支路，通道数都变为1/2，合并后为2，再减半后为1
        out = self.fuse(torch.cat([self.branch[0](x), self.branch[1](out1), self.branch[2](out2), self.branch[3](out3)], dim=1))
        out = self.att(out)     # ESA 模块
        out += x                # 实现输出的残差
        return out


# EFDN 网络（超分倍数4，输入通道3，中间通道48，输出通道3）    长宽各放大四倍
class EFDN(nn.Module):
    def __init__(self, scale=4, in_channels=3, n_feats=48, out_channels=3):
        super(EFDN, self).__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)            # 3*3卷积 Size 不变  通道数增加
        
        # 主体模块 —— 中间层四个 EFDB 主要模块
        self.cells = nn.ModuleList([Cell(n_feats) for _ in range(4)])
        # 中间融合 —— 三个1*1卷积模块  Size不变，通道数减半
        self.local_fuse = nn.ModuleList([nn.Conv2d(n_feats*2, n_feats, 1, 1, 0) for _ in range(3)])

        # 最终融合 —— Pixel Shuffle 
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, out_channels*(scale**2), 3, 1, 1),       # 1*1卷积 Size 不变 通道数变为输出通道数的 scale*scale 倍
            nn.PixelShuffle(scale)                                      # 像素重排 
        )

    def forward(self, x):
        # 头部  通道数增加
        out0 = self.head(x)

        # 主体
        out1 = self.cells[0](out0)
        out2 = self.cells[1](out1)
        out2_fuse = self.local_fuse[0](torch.cat([out1, out2], dim=1))
        out3 = self.cells[2](out2_fuse)
        out3_fuse = self.local_fuse[1](torch.cat([out2, out3], dim=1))
        out4 = self.cells[3](out3_fuse)
        out4_fuse = self.local_fuse[2](torch.cat([out2, out4], dim=1))

        out = out4_fuse + out0      # 实现最外层的残差

        # 末尾  最终融合
        out = self.tail(out)

        return out.clamp(0,1)


# -------------------------------------------------
# 以下部分是 unitv2.py 中实现的模块

# 将不同尺寸的卷积核 通过对称补0的方式 填充成相同大小
def  multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])


# EDBB中可配置的双层卷积分支（序列类型，输入输出通道数，两卷积之间通道数控制参数）
# 支持多种组合类型，训练时使用多分支结构，推理时合并为单个等效卷积层
class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type            # 序列类型
        self.inp_planes = inp_planes    # 输入通道数
        self.out_planes = out_planes    # 输出通道数

        # 根据不同分支类型，使能不同双层卷积分支，并提取参数
        # 1*1-3*3卷积分支
        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
        
        # 1*1-sobel算子x方向，水平方向卷积核——只可学习缩放因子与学习率
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # 初始化可学习的缩放因子和偏置
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3   # 可学习缩放因子，每个通道1个数
            self.scale = nn.Parameter(scale)    # 注册为可学习参数
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # 初始化掩码（固定卷积核）
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32) # 输入1通道输出out_planes通道的3*3卷积核
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False) # 将张量包装成模型参数且不可学习

        # 1*1-sobel算子y方向——同上
        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        # 1*1卷积-拉普拉斯算子——同上
        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    # 前向传播方式——传入特征图x，输出y1
    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)  # 加载权重与偏置进行卷积
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)     # 填充一圈像素，为3*3卷积打基础
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad                       # 将填充后的边缘替换为偏置，其他位置不变
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1) # 加载权重3*3卷积
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3 只有这里与上面不一样，权重不是可学习参数，缩放因子scale才是
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1
    
    # 这是重参数化方法
    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:      # 如果是CPU
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))  # 合并卷积核——卷积 w0*w1
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1  # 合并偏置——w1*b0+b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]    # 构造对角卷积核
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3)) # 合并卷积核——卷积 w0*w1
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1   # 合并偏置——w1*b0+b1
        return RK, RB


# 训练时的 EDBB 模块（输入输出通道数、1*1-1*3多层结构、激活函数、、导出模式、）
class EDBB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier=None, act_type='prelu', with_idt = False, deploy=False, with_13=False, gv=False):
        super(EDBB, self).__init__()
        
        self.deploy = deploy            # 是否导出模式
        self.act_type = act_type        # 激活函数类型
        
        self.inp_planes = inp_planes    # 输入输出通道数
        self.out_planes = out_planes

        self.gv = gv          

        if depth_multiplier is None:    # 如果不使用多倍内部通道
            self.depth_multiplier = 1.0
        else: 
            self.depth_multiplier = depth_multiplier   # 对于 mobilenet，最好有 2X 内部通道
        
        if deploy:                      # 导出模式就是1个3*3的带偏置卷积
            self.rep_conv = nn.Conv2d(in_channels=inp_planes, out_channels=out_planes, kernel_size=3, stride=1,padding=1, bias=True)
        else:                           # 训练时的一般模式
            self.with_13 = with_13
            if with_idt and (self.inp_planes == self.out_planes):
                self.with_idt = True    # 必须是输入输出通道数相同，with_idt才有效
            else:
                self.with_idt = False

            # EDBB 中的六条分支
            self.rep_conv = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
            self.conv1x1 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
            self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
            self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
            self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.deploy:     # 导出模式
            y = self.rep_conv(x)
        elif self.gv:       # gv模式
            y = self.rep_conv(x)     + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x) + x 
        else:
            y = self.rep_conv(x)     + \
                self.conv1x1(x)     + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x) 
            if self.with_idt:           # 可选残差
                y += x
            if self.with_13:            # 可选1*1-3*3
                y += self.conv1x1_3x3(x)

        if self.act_type != 'linear':
            y = self.act(y)
        return y
    
    # 将一般的模型结构重参数化成 gv模式
    def switch_to_gv(self):
        if self.gv:     # 确保只在首次调用时执行合并操作，避免重复计算。
            return
        self.gv = True
        
        K0, B0 = self.rep_conv.weight, self.rep_conv.bias   # 3*3卷积参数
        K1, B1 = self.conv1x1_3x3.rep_params()              # 1*1-3*3参数
        K5, B5 = multiscale(self.conv1x1.weight,3), self.conv1x1.bias   # 1*1参数
        RK, RB = (K0+K5), (B0+B5)                           # 先融合 3*3 与 1*1
        if self.with_13:                                    # with_13 控制是否融合 1*1 - 3*3
            RK, RB = RK + K1, RB + B1

        self.rep_conv.weight.data = RK      # 将融合后的参数与权重复制
        self.rep_conv.bias.data = RB
        
        for para in self.parameters():      # 断开所有参数的梯度计算，固定合并后的参数，确保切换后仅用于推理，不再参与训练更新。
            para.detach_()
      
    
    # 将一般的模型结构直接重参数化成 deploy模式
    def switch_to_deploy(self):
        if self.deploy:     # 同上
            return
        self.deploy = True
        
        K0, B0 = self.rep_conv.weight, self.rep_conv.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        K5, B5 = multiscale(self.conv1x1.weight,3), self.conv1x1.bias
        if self.gv:
            RK, RB = (K0+K2+K3+K4),(B0+B2+B3+B4) # gv 导出
        else:
            RK, RB = (K0+K2+K3+K4+K5), (B0+B2+B3+B4+B5) # 一般模式导出
            if self.with_13:
                RK, RB = RK + K1, RB + B1   # 带有 1*1-3*3
        if self.with_idt:                   # 带残差
            device = RK.get_device()
            if device < 0:                  # CPU
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0     # 偏置权重1偏移0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt        
            
        # 重新创建新的网络结构，并加载参数
        self.rep_conv = nn.Conv2d(in_channels=self.inp_planes, out_channels=self.out_planes, kernel_size=3, stride=1,padding=1, bias=True)
        self.rep_conv.weight.data = RK
        self.rep_conv.bias.data = RB
        
        for para in self.parameters():  # 断开所有参数的梯度计算，固定合并后的参数，确保切换后仅用于推理，不再参与训练更新。
            para.detach_()
            
        #self.__delattr__('conv3x3')    # 删除不存在的属性
        self.__delattr__('conv1x1_3x3')
        self.__delattr__('conv1x1')
        self.__delattr__('conv1x1_sbx')
        self.__delattr__('conv1x1_sby')
        self.__delattr__('conv1x1_lpl')


# 导出的 EDBB 模块（输入通道数，输出通道数）
class EDBB_deploy(nn.Module):
    def __init__(self, inp_planes, out_planes):
        super(EDBB_deploy, self).__init__()
        # 等效成不改变 Size 的 3*3 卷积
        self.rep_conv = nn.Conv2d(in_channels=inp_planes, out_channels=out_planes, kernel_size=3, stride=1,padding=1, bias=True)
        self.act = nn.PReLU(num_parameters=out_planes) # 激活函数

    def forward(self, x):
        y = self.rep_conv(x)
        y = self.act(y)
        
        return y
# -------------------------------------------------
