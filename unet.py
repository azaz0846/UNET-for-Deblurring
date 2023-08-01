import gc
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
from util import *
from skimage import metrics
from ssim import *
class FishDataset(Dataset):
    def __init__(self,x ,y=None):
        super(FishDataset).__init__()

        self.x = x
        self.y = y

    def __getitem__(self, index):
        
        if self.y is None:
            return self.x[index][0]
        else:
            return self.x[index][0], self.y[index][0]

    def __len__(self):
        return len(self.x)
        

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Concatenate_different_channels(nn.Module):
    """([BN] => ReLU => 1*1cov => [BN] => ReLU => 3*3cov)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels
        self.concat_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1, bias=False),
            #nn.Conv2d(in_channels, 2 * out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )


    def forward(self, x):
        return self.concat_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.concat_cov = Concatenate_different_channels(480, in_channels // 2) #down1~4 channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.silu = nn.SiLU()
        #self.sa = SpatialAttention()
        ##self.Attention = Attention(weight_visialize=True)

    def forward(self, x, x1, x2, x3, x4):
        x = self.up(x)
        """
        se1 = SELayer(x1.shape[1]).to('cuda')
        se2 = SELayer(x2.shape[1]).to('cuda')
        se3 = SELayer(x3.shape[1]).to('cuda')
        se4 = SELayer(x4.shape[1]).to('cuda')
        x1 = se1(x1)
        x2 = se2(x2)
        x3 = se3(x3)
        x4 = se4(x4)
        """
        """
        #x = self.silu(x)
        #x1 = self.silu(x1)
        #x2 = self.silu(x2)
        #x3 = self.silu(x3)
        #x4 = self.silu(x4)
        ca1 = ChannelAttention(x1.shape[1]).to('cuda')
        ca2 = ChannelAttention(x2.shape[1]).to('cuda')
        ca3 = ChannelAttention(x3.shape[1]).to('cuda')
        ca4 = ChannelAttention(x4.shape[1]).to('cuda')
        x1 = x1 * ca1(x1)
        x2 = x2 * ca2(x2)
        x3 = x3 * ca3(x3)
        x4 = x4 * ca4(x4)
        #x1 = x1 * self.sa(x1)
        #x2 = x2 * self.sa(x2)
        #x3 = x3 * self.sa(x3)
        #x4 = x4 * self.sa(x4)
        """
        """
        # embedding + attention
        n_batchsize, Chann, width ,height = x1.shape
        Attnx1 = Self_Attn(x1.shape[1]).to('cuda')
        Attnx2 = Self_Attn(x2.shape[1]).to('cuda')
        Attnx3 = Self_Attn(x3.shape[1]).to('cuda')
        Attnx4 = Self_Attn(x4.shape[1]).to('cuda')
        x1 = Attnx1(x1)
        x2 = Attnx2(x2)
        x3 = Attnx3(x3)
        x4 = Attnx4(x4)
        x = F.interpolate(x, size=(width, height))
        x1 = F.interpolate(x1, size=(width, height))
        x2 = F.interpolate(x2, size=(width, height))
        x3 = F.interpolate(x3, size=(width, height))
        x4 = F.interpolate(x4, size=(width, height))
        #print(x.shape)
        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)
        #print(x4.shape)
        """
        """
        # patch embedding + mult attention
        embedding1 = Embeddings(img_size=x1.size()[2],in_channels=x1.size()[1]).to('cuda')
        embedding2 = Embeddings(img_size=x2.size()[2],in_channels=x2.size()[1]).to('cuda')
        embedding3 = Embeddings(img_size=x3.size()[2],in_channels=x3.size()[1]).to('cuda')
        embedding4 = Embeddings(img_size=x4.size()[2],in_channels=x4.size()[1]).to('cuda')
        x1 = embedding1(x1)
        x2 = embedding2(x2)
        x3 = embedding3(x3)
        x4 = embedding4(x4)
        x1, we1 =self.Attention(x1)
        x2, we2 =self.Attention(x2)
        x3, we3 =self.Attention(x3)
        x4, we4 =self.Attention(x4) 
        
        width, height = x.shape[2], x.shape[3]
        #x = F.interpolate(x, size=(width, height))
        x1 = F.interpolate(x1, size=(width, height))
        x2 = F.interpolate(x2, size=(width, height))
        x3 = F.interpolate(x3, size=(width, height))
        x4 = F.interpolate(x4, size=(width, height))
        #print(x.shape)
        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)
        #print(x4.shape)
        # input is CHW
        #diffY = x2.size()[2] - x1.size()[2]
        #diffX = x2.size()[3] - x1.size()[3]
        """
        #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])
        x = F.interpolate(x, size=(x1.size()[2], x1.size()[3]), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(x1.size()[2], x1.size()[3]), mode='bilinear', align_corners=True)
        #c = torch.cat([x4, x3, x2, x1], dim=1)
        """
        b1 = nn.BatchNorm2d(x1.size()[1]).to('cuda')
        b2 = nn.BatchNorm2d(x2.size()[1]).to('cuda')
        b3 = nn.BatchNorm2d(x3.size()[1]).to('cuda')
        b4 = nn.BatchNorm2d(x4.size()[1]).to('cuda')
        
        x = b1(x)
        x1 = b1(x1)
        x2 = b2(x2)
        x3 = b3(x3)
        x4 = b4(x4)
        """
        pre_x = torch.cat([x1, x2, x3, x4], dim=1)
        #se = SELayer(x.shape[1]).to('cuda')
        #x = se(x)
        
        #concat_cov = Concatenate_different_channels(pre_x.shape[1], x.shape[1]).to('cuda')
        #concat_cov = Concatenate_different_channels(self.in_channels, self.out_channels).to('cuda')
        pre_x = self.concat_cov(pre_x)
        
        x = torch.cat([pre_x, x], dim=1)
        ##dc = DoubleConv(x.shape[1], self.out_channels, self.out_channels // 2).to('cuda')
        ## = dc(x)
        
        #x = torch.cat([x1, x], dim=1)
        #outc = OutConv(x.shape[1], self.in_channels).to('cuda')
        #x = outc(x)
        
        #ca = ChannelAttention(x.shape[1]).to('cuda')
        #x = x * ca(x)
        
        
        #return pre_x
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.batchn = nn.BatchNorm2d(n_classes)

    def forward(self, x):
        #origin = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x1, x2, x3, x4)
        x = self.up2(x, x1, x2, x3, x4)
        x = self.up3(x, x1, x2, x3, x4)
        x = self.up4(x, x1, x2, x3, x4)
        logits = self.outc(x)
        #logits = logits + origin
        #logits = torch.cat([logits, logits, logits], dim=1)
        return torch.sigmoid(logits)

def train_net(model, config, device):

    
    
    #inputdata = ImageFolder('./train/', transform=transforms.Compose([transforms.ToTensor()]))
    #targetdata = ImageFolder('./label/', transform=transforms.Compose([transforms.Resize(config['batch_size'],3,360,360),transforms.ToTensor()]))
    X_train, X_valid, y_train, y_valid = [], [], [], []
    
    # 1. Create dataset
    X_train = ImageFolder(f'./dataset/train/', transform=transforms.ToTensor())   #output [Image, class]; we need inputdata[i][0] to get image;inputdata[i][1] is the class No.
    y_train = ImageFolder(f'./dataset/label/train', transform=transforms.ToTensor())
    X_valid = ImageFolder(f'./dataset/test/', transform=transforms.ToTensor())
    y_valid = ImageFolder(f'./dataset/label/test', transform=transforms.ToTensor())

    #inputdata=transforms.Resize(inputdata.size()[0],inputdata.size()[1],inputdata.size()[2]/2,inputdata.size()[3]/2)(inputdata)
    #targetdata = transforms.Resize(targetdata.size()[0],targetdata.size()[1],targetdata.size()[2]/2,targetdata.size()[3]/2)(targetdata)
    #print(inputdata.size(),targetdata.size())
    # 2. Split train & validation

    #X_train, X_valid, y_train, y_valid = train_test_split(inputdata, targetdata, test_size=config['valid_ratio'])


    print('X_train:{}, X_valid:{}, y_train:{}, y_valid:{}', len(X_train), len(X_valid), len(y_train), len(y_valid))
    #print(targetdata[0])

    train_dataset = FishDataset(X_train, y_train)
    valid_dataset = FishDataset(X_valid, y_valid)
    
    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)
    
    
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)  # goal: maximize Dice score
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
    ##                       min_lr=1e-4)
    #判斷能否使用自動混合精度
    enable_amp = True if "cuda" in device.type else False
    print('amp: ', enable_amp)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    #####################
    #criterion = nn.CrossEntropyLoss() #loss function
    #criterion = DiceBCELoss()  #Segmentation loss
    ###criterion = MS_SSIM_L1_LOSS() # deblur loss
    ##criterion = SSIM().cuda()
    criterion = MS_SSIM(data_range=1, size_average=True, channel=3) # reuse the gaussian kernel with SSIM & MS_SSIM. 
    #####################


    #use < tensorboard --logdir=UNet --bind_all > comment in the terminal to display loss and acc
    writer = SummaryWriter("UNet") # Writer of tensoboard.
    step = 0
    best_acc = 0
    last_loss = 100
    patience = 5
    triggertimes = 0
    path = 'output.txt'
    mean_train_loss_record = []
    #把tensor image二值化
    trans1 = transforms.Compose([transforms.Grayscale(num_output_channels=1),#轉灰度圖 channel=1
                        torch.sigmoid, 
                        ThresholdTransform(thr_255=128)  #上面有定義class Threshold 0~255
                        ])
    trans2 = transforms.Compose([torch.sigmoid, #轉灰度圖 channel=1
                        ThresholdTransform(thr_255=128)  #上面有定義class Threshold 0~255                
                        ])
    #for x, y in train_loader:
        
    for epoch in range(config['n_epochs']):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()# Set model to train mode.
        loss_record = []
        acc_record = []
        # tqdm is a package to visualize training progress.
        train_pbar = tqdm(train_loader)
        #realsizex, realsizey = 0
        for x, y in train_pbar:
            #realsizex, realsizey = x.shape[2], x.shape[3]
            #reduce height & weight size
            #x = F.interpolate(x, scale_factor=(0.2,0.2))
            #y = F.interpolate(y, scale_factor=(0.2,0.2)) 
            x = F.interpolate(x, size=(224,224))
            y = F.interpolate(y, size=(224,224))
            ##y = trans1(y)  #transformes二值化target圖片
            
            x, y = x.to(device), y.to(device) # Move data to device.
            #前向過程 model + loss 開啟 autocast
            ##with torch.cuda.amp.autocast(enabled=enable_amp):
        
            #autocast上下文應該只包含網絡的前向過程（包括loss的計算），而不要包含反向傳播。
            optimizer.zero_grad() # Set gradient to zero.
            pred = model(x)
            ###loss = criterion(pred, y)
            
            #print('\n', x.max().item(), x.min().item(), y.max().item(), y.min().item(), pred.max().item(),pred.min().item(), sep=' | ')
            
            loss = 1 - criterion(pred, y)
            loss.backward() # Compute gradient(backpropagation).
            optimizer.step() # Update parameters.
            ##grad_scaler.scale(loss).backward()
            ##grad_scaler.step(optimizer)
            ##grad_scaler.update()
            #count accuracy
            acc = ms_ssim(y, pred, data_range = 1)
            #acc = compute_psnr(y, pred, )
            step += 1
            loss_record.append(loss.item())
            acc_record.append(acc.item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{config["n_epochs"]}]')
            train_pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
        
        mean_train_loss = sum(loss_record)/len(loss_record)
        mean_train_acc = sum(acc_record) / len(acc_record)
        writer.add_scalar('train_loss/step', mean_train_loss, step)
        writer.add_scalar('train_acc/step', mean_train_acc, step)


        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval() # Set your model to evaluation mode.
        loss_record = []
        acc_record = []
        tmp_pred = torch.zeros(1)
        tmp_y = torch.zeros(1)
        tmp_x = torch.zeros(1)
        tmp_acc = 0
        for x, y in valid_loader:
            ##y = trans1(y)  #transformes二值化target圖片
            #x = F.interpolate(x, scale_factor=(0.2,0.2))
            #y = F.interpolate(y, scale_factor=(0.2,0.2))
            #save_image(x, f'./predict/valid_x_{epoch}.png')
            #save_image(y, f'./predict/valid_y_{epoch}.png')
            x = F.interpolate(x, size=(224,224))
            y = F.interpolate(y, size=(224,224))
            x, y =x.to(device=device), y.to(device=device)
            with torch.no_grad():
                pred = model(x)
                loss = 1 - criterion(pred, y)
                #count accuracy
                #pred = trans2(pred)  #transformes二值化prediction圖片
                acc = ms_ssim(y, pred, data_range = 1)
                #acc = metrics.structural_similarity(y, pred, multichannel=True)
                loss_record.append(loss.item())
                acc_record.append(acc.item())
                #save_image(pred, f'./predict/valid_pred_{epoch}.png')
                tmp_pred = pred
                tmp_acc = acc.item()
            tmp_x = x
            tmp_y = y
            

        
        #print('\n', tmp_x.max().item(), tmp_x.min().item(), tmp_y.max().item(), tmp_y.min().item(), tmp_pred.max().item(),tmp_pred.min().item(), sep=' | ')
        
        save_image(tmp_pred, './predict/valid_pred_{}_SSIM_{:.4f}.png'.format(epoch+1, tmp_acc))
        save_image(tmp_y, f'./predict/valid_y_{epoch+1}.png')
        save_image(tmp_x, f'./predict/valid_x_{epoch+1}.png')
        mean_valid_loss = sum(loss_record)/len(loss_record)
        scheduler.step(mean_valid_loss)
        mean_valid_acc = sum(acc_record) / len(acc_record)
        print(f'Epoch [{epoch+1}/{config["n_epochs"]}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}, Train accuracy: {mean_train_acc:.4f}, Valid accuracy: {mean_valid_acc:.4f}')
        with open(path, 'a') as f:
            print(f'Epoch [{epoch+1}/{config["n_epochs"]}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}, Train accuracy: {mean_train_acc:.4f}, Valid accuracy: {mean_valid_acc:.4f}', file=f)
        writer.add_scalar('valid_loss/step', mean_valid_loss, step)
        writer.add_scalar('valid_acc/step', mean_valid_acc, step)
        #print(f'Epoch [{epoch+1}/{config["n_epochs"]}]: Train loss: {mean_train_loss:.4f}, Train accuracy: {mean_train_acc:.4f}, Valid loss: {mean_valid_loss:.4f}, Valid accuracy: {mean_valid_acc:.4f}')
        if mean_valid_acc > best_acc:
            print(f"Best model found at epoch {epoch+1}, saving model")
            torch.save(model.state_dict(), f"./models/UNet_best_at_epoch_{epoch+1}.ckpt") # only save best to prevent output memory exceed error
            best_acc = mean_valid_acc

        #Early stopping
        current_loss = mean_train_loss
        mean_train_loss_record.append(mean_train_loss)
        if (last_loss - current_loss) <= 0.0001: # 紀錄五步前的loss並計算是否相差超過0.0001
            triggertimes += 1
            print('Trigger Times:', triggertimes)

            if triggertimes >= patience:
                print('Early stopping!')
                torch.save(model.state_dict(), f"./models/UNet_Early_stop_at_epoch_{epoch+1}.ckpt")
                return

        else:
            print('trigger times: 0')
            triggertimes = 0

        ##last_loss = current_loss
        last_loss = mean_train_loss_record[(epoch + 1) - 5] if epoch + 1 >= 5 else last_loss # 紀錄五步前的loss
        print('last_loss: ', last_loss)
        #gc.collect() 
        #torch.cuda.empty_cache()
    writer.close()
    
    torch.save(model.state_dict(), f"./models/UNet_{config['n_epochs']}_epoch.ckpt")
        
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = UNet(n_channels=3, n_classes=3, bilinear=True)
    net.to(device=device)
    config = {
        'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
        'n_epochs': 200,     # Number of epochs.            
        'batch_size': 4, 
        'learning_rate': 1e-3,               
    }
    try:
        train_net(net, config, device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(),'./models/INTERRUPTED.pth')
        print('Saved interrupt')
        raise



"""
class double_convolution(nn.Module):
    #(convolution => [BN] => ReLU) * 2
    def __init__(self, in_channels, out_channels):
        super(double_convolution,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    #Downscaling with maxpool then double conv

    def __init__(self, in_channels, out_channels):
        super(Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_convolution(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    #Upscaling then double conv
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = double_convolution(in_channels, out_channels)
        self.in_channels = in_channels

    def forward(self, x, x1):
        x = self.up(x)
        #tensor.size(): [minibatchsize, channels, H, W]
        #將x2 size CenterCrop(中心裁切)成x1 size才能拼接一起
        #x2 = transforms.CenterCrop(
        #    (x1.size()[2], x1.size()[3]))(x2)
        #x1 = transforms.CenterCrop((x.size()[2], x.size()[3]))(x1)
        #x2 = transforms.CenterCrop((x.size()[2], x.size()[3]))(x2)
        #x3 = transforms.CenterCrop((x.size()[2], x.size()[3]))(x3)
        #x4 = transforms.CenterCrop((x.size()[2], x.size()[3]))(x4)
        
        #offset = x1.size()[2] - x2.size()[2]
        #padding = 2 * [offset // 2, offset // 2]
        #x2 = F.pad(x2, padding)
        #x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]))
        x1 = F.interpolate(x1, size=(x.size()[2], x.size()[3]))
        #x2 = F.interpolate(x2, size=(x.size()[2], x.size()[3]))
        #x3 = F.interpolate(x3, size=(x.size()[2], x.size()[3]))
        #x4 = F.interpolate(x4, size=(x.size()[2], x.size()[3]))

        x = torch.cat([x1, x], dim=1)
        #pointwise = nn.Conv2d(x.shape[1], self.in_channels, kernel_size=1).to('cuda')
        #x = pointwise(x)
        return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv_block = double_convolution(3, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        self.nonlocal1 = NONLocalBlock2D(in_channels=32, inter_channels=32 // 4)
        self.conv_block1 = double_convolution(32, 64)
        self.nonlocal2 = NONLocalBlock2D(in_channels=64, inter_channels=64 // 4)
        self.conv_block2 = double_convolution(64, 128)
    def forward(self, x):
        x1 = self.conv_block(x)

        #x1 = self.maxpool(x1)
        #x1 = self.nonlocal1(x1)
        #x2 = self.conv_block1(x1)
        x2 = self.down1(x1)
        
        #x2 = self.maxpool(x2)
        #x2 = self.nonlocal2(x2)
        #x3 = self.conv_block2(x2)
        x3 = self.down2(x2)

        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x
"""