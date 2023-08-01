import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from scipy import signal
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 压缩空间
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  # [b, C, 1, 1]
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 压缩通道
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # 压缩通道
        x = torch.cat([avg_out, max_out], dim=1)  # [b, 1, h, w]
        x = self.conv1(x)
        return self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type


class _NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=3,
                 sub_sample=True,
                 bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)

        print(f.shape)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, 
                                              bn_layer=bn_layer)

#########loss##################

#loss function
#https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #print(inputs.shape, targets.shape)
        targets = F.interpolate(targets, size=(inputs.size()[2],inputs.size()[3]))  #resize h, w
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        #targets = torch.sigmoid(targets)   
        #flatten label and prediction tensors
        #print(inputs.shape, targets.shape)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

#alternative loss function
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice



#deblur loss
#source:https://github.com/psyrocloud/MS-SSIM_L1_LOSS
class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()
        


#########accuracy################

##deblur evaluation
#source:https://blog.csdn.net/bby1987/article/details/109373572
def compute_mse(X, Y):
    """
    #MSE越小表明待評估的圖像品質越好。
    compute mean square error of two images

    Parameters:
    X, Y: numpy array
        two images data

    Returns:
    mse:float
        mean square error
    """
    X = torch.float32(X)
    Y = torch.float32(Y)
    mse = torch.mean((X - Y) ** 2, dtype=torch.float64) # L2 loss
    return mse

def compute_psnr(X, Y, data_range):
    """
    compute peak signal to noise ratio of two images
    #由於PSNR公式將MSE放在了分母上，所以在進行圖像品質評估時，PSNR數值越大表明待評估的圖像品質越好，這一點與MSE相反。
    
    Parameters:
    X, Y: numpy array
        two images data

    Returns:
    psnr: float
        peak signal to noise ratio
    """
    mse = compute_mse(X, Y)
    psnr = 10 * torch.log10((data_range ** 2)/ mse)
    return psnr

def compute_ssim(X, Y, win_size = 7, data_range = None):
    """
    compute structural similarity of two images

    Parameters:
    X, Y: numpy array
        two images data
    win_size: int
        window size of image patch for computing ssim of one single position
    data_range: int or float
        maximum dynamic range of image data type

    Returns:
    mssim: float
        mean structural similarity
    ssim_map: numpy array (float)
        structural similarity map, same shape as input images
    """
    assert X.shape == Y.shape, "X, Y must have same shape"
    assert X.dtype == Y.dtype, "X, Y must have same dtype"
    assert win_size <= np.min(X.shape[2:4]), "win_size should be <= shorter edge of image"
    assert win_size % 2 == 1, "win_size must be odd"
    if data_range is None:
        if 'float' in str(X.dtype):
            data_range = 1
        elif 'uint8' in str(X.dtype):
            data_range = 255
        else:
            raise ValueError('image dtype must be uint8 or float when data_range is None')

    X = np.squeeze(X)
    Y = np.squeeze(Y)
    if X.ndim == 2:
        mssim, ssim_map = _ssim_one_channel(X, Y, win_size, data_range)
    elif X.ndim == 4:
        ssim_map = np.zeros(X.shape)
        for i in range(X.shape[3]):
            _, ssim_map[:, :, :, i] = _ssim_one_channel(X[:, :, :, i], Y[:, :, :, i], win_size, data_range)
        mssim = np.mean(ssim_map)
    else:
        raise ValueError("image dimension must be 2 or 3")
    return mssim, ssim_map

def _ssim_one_channel(X, Y, win_size, data_range):
    """
    compute structural similarity of two single channel images

    Parameters:
    X, Y: numpy array
        two images data
    win_size: int
        window size of image patch for computing ssim of one single position
    data_range: int or float
        maximum dynamic range of image data type

    Returns:
    mssim: float
        mean structural similarity
    ssim_map: numpy array (float)
        structural similarity map, same shape as input images
    """
    X ,Y = normalize(X, Y, data_range)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    num = win_size ** 2
    kernel = torch.ones([win_size, win_size]) / num
    mean_map_x = convolve2d(X, kernel)
    mean_map_y = convolve2d(Y, kernel)

    mean_map_xx = convolve2d(X * X, kernel)
    mean_map_yy = convolve2d(Y * Y, kernel)
    mean_map_xy = convolve2d(X * Y, kernel)

    cov_norm = num / (num - 1)
    var_x = cov_norm * (mean_map_xx - mean_map_x ** 2)
    var_y = cov_norm * (mean_map_yy - mean_map_y ** 2)
    covar_xy = cov_norm * (mean_map_xy - mean_map_x * mean_map_y)

    A1 = 2 * mean_map_x * mean_map_y + C1
    A2 = 2 * covar_xy + C2
    B1 = mean_map_x ** 2 + mean_map_y ** 2 + C1
    B2 = var_x + var_y + C2

    ssim_map = (A1 * A2) / (B1 * B2)
    mssim = np.mean(ssim_map)
    return mssim, ssim_map

def normalize(X, Y, data_range):
    """
    convert dtype of two images to float64, and then normalize them by data_range

    Parameters:
    X, Y: numpy array
        two images data
    data_range: int or float
        maximum dynamic range of image data type
    
    Returns:
    X, Y: numpy array
        two images
    """
    X = X.float() / data_range
    Y = Y.float() / data_range
    return X, Y

def convolve2d(image, kernel):
    """
    convolve single channel image and kernel

    Parameters:
    image: numpy array
        single channel image data
    kernel: numpy array
        kernel data

    Returns:
    result: numpy array
        image data, same shape as input image
    """
    result = signal.convolve2d(image.cpu().detach().numpy(), kernel.cpu().detach().numpy(), mode = 'same', boundary = 'fill')
    return result

#source:https://blog.csdn.net/hyk_1996/article/details/87867285
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel = 3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size = 11, window = None, size_average = True, full = False, val_range = None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to('cuda')
    window = window.to('cuda')
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2) # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average=True, val_range = None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        #Assume 3 channel for SSIM
        self.channel = 3
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input, target, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def evaluate(mask_pred, mask_true, net):

    # convert to one-hot format
    if net.n_classes == 1:
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
        # compute the Dice score
        return dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
    else:
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        return multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

def iou_score(target, prediction):
    #prediction = transforms.Grayscale(num_output_channels=1)(prediction)
    #target = transforms.Grayscale(num_output_channels=1)(target)
    #prediction = torch.sigmoid(prediction) 
    #print(prediction)
    #prediction[0] = torch.where(prediction[0]>0,1,0)
    """
    trans1 = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                        ThresholdTransform(thr_255=128)
                        ])
    prediction = trans1(prediction)
    target = trans1(target)
    """
    #save_image(prediction, f'./predict/prediction1.png')
    #print(prediction[0])
    #save_image(prediction, f'./predict/prediction.png')
    #print('prediction[0] ', prediction[0])
    #print(target)
    #save_image(target, f'./predict/target1.png')
    #print('target.shape ', target.shape)
    target = F.interpolate(target, size=(prediction.size()[2],prediction.size()[3]))
    #inputs = torch.sigmoid(inputs)       
    #targets = torch.sigmoid(targets)  

    intersection = torch.logical_and(target, prediction)
    union = torch.logical_or(target, prediction)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embedded_dim,
    dropout=0.1):
        super().__init__()
        self.patch_embedded = nn.Conv2d(in_channels=in_channels, out_channels=embedded_dim, kernel_size = patch_size, stride = patch_size, bias =False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # X = [batchsize, 1, 28, 28]
        x = self.patch_embedded(x)
        # X = [batchsize, embedded_dim, h, w]
        #x = x.flatten(2)
        # X = [batchsize, embedded_dim, h*w]
        #x = x.transpose(2, 1)
        # X = [batchsize, h*w, embedded_dim]
        #x = self.dropout(x)
        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        #patch_embedded
        patch_size = 16
        self.patch_embedded = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size = patch_size, stride = patch_size, bias =False)

        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        x = self.patch_embedded(x)
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B*N*C/8
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B*C*N/8
        energy =  torch.bmm(proj_query,proj_key) # batch的matmul B*N*N
        attention = self.softmax(energy) # B * (N) * (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1, width*height) # B * C * N
 
        out = torch.bmm(proj_value,attention.permute(0,2,1) ) # B*C*N
        out = out.view(m_batchsize,C,width,height) # B*C*H*W
 
        out = self.gamma*out + x
        return out

class Embeddings(nn.Module):

    def __init__(self, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = img_size #224
        patch_size = 16 if img_size >= 16 else 1
        hidden_size = 128
        #將圖片分割成多少塊 (224/16) * (224/16) = 196
        n_patches = (img_size//patch_size) * (img_size//patch_size)
        #對圖片進行卷積獲取圖片的塊，將每塊映射成768維
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)

        #設置可學習的位置編碼訊息 (1, 196+1, 768)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size)) #nn.Parameter(torch.zeros(1, n_patches+1, hidden_size))

        #設置可學習的分類訊息維度
        self.classifer_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        bs = x.shape[0]
        #cls_tokens = self.classifer_token.expand(bs, -1, -1)(bs, 1, 768)
        x = self.patch_embeddings(x) # (bs, 768, 14, 14)
        x = x.flatten(2) #(bs, 768, 196)
        x = x.transpose(-1, -2) #(bs, 196, 768)
        ##x= torch.cat((cls_tokens, x), dim=1) #分類訊息與圖片塊進行拼接 (bs, 197, 768)
        embeddings = x + self.position_embeddings # 將圖片塊訊息和對其位置訊息進行相加(bs, 197, 768)
        embeddings = self.dropout(embeddings)
        return embeddings

class Attention(nn.Module):
    def __init__(self, weight_visialize):
        super(Attention, self).__init__()
        self.weight_visialize = weight_visialize
        self.num_attention_heads = 2 #12
        self.hidden_size = 128
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads) #768/12 = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 12*64 = 768

        self.query = nn.Linear(self.hidden_size, self.all_head_size) #768 -> 768 Wq = (768, 768)
        self.key = nn.Linear(self.hidden_size, self.all_head_size) #768->768 Wk = (768, 768)
        self.value = nn.Linear(self.hidden_size, self.all_head_size) #768->768 Wv = (768, 768)
        self.out = nn.Linear(self.hidden_size, self.hidden_size) #768->768
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)

        self.softmax = nn.Softmax(dim = -1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) #(bs, 197)+(12, 64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (bs, 12, 197, 64)

    def forward(self, hidden_states):
        # hidden_states: (bs, 197, 768)
        mixed_query_layer = self.query(hidden_states) # 768->768
        mixed_key_layer = self.key(hidden_states) # 768->768
        mixed_value_layer = self.value(hidden_states) # 768->768

        query_layer = self.transpose_for_scores(mixed_query_layer) # (bs, 12, 197, 64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #將q向量與k向量進行相乘 (bs, 12, 197, 197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) #除以向量維度開方
        attention_probs = self.softmax(attention_scores) #將分數轉成機率
        weights = attention_probs if self.weight_visialize else None # 將attention權重輸出
        attention_probs =self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) #將機率與value向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # (bs, 197)+(768,)=(bs, 197, 768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        #return attention_output, weights # (bs, 197, 768), (bs, 197, 197)
        attention_output = attention_output.permute(0, 2, 1)
        a_b, a_c, a_n = attention_output.shape
        attention_output = attention_output.view(a_b, a_c, int(math.sqrt(a_n)), int(math.sqrt(a_n)))#int(math.sqrt(a_n))
        return attention_output, weights # (bs, 768, 14, 14), (bs, 196, 196) 少了class token

