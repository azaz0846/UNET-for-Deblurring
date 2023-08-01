from unet import *
from torchvision.utils import save_image
from skimage import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
testdata = ImageFolder('./test/', transform=transforms.ToTensor())
labeldata = ImageFolder('./testlabel/', transform=transforms.ToTensor())
#test_dataset = FishDataset(testdata)
test_dataset = FishDataset(testdata, labeldata)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model_best = UNet(n_channels=3,n_classes=3,bilinear=True).to(device)
model_best.load_state_dict(torch.load("./models/UNet_best_at_epoch_29.ckpt"))
model_best.eval()
prediction = []

with torch.no_grad():
    for i, (data, label) in enumerate(test_loader):
        origshape = data.shape
        data = F.interpolate(data, size=(224,224), mode='bilinear', align_corners=True)
        test_pred = model_best(data.to(device))
        test_pred = F.interpolate(test_pred, size=(origshape[2],origshape[3]), mode='bilinear', align_corners=True)
        #acc = ms_ssim(data.to(device), torch.sigmoid(test_pred))
        acc = ms_ssim(label.to(device), test_pred, data_range = 1)
        #psnr = compute_psnr(torch.sigmoid(test_pred), label.to(device), data_range=1)
        psnr = metrics.peak_signal_noise_ratio(label.cpu().numpy(), test_pred.cpu().numpy(), data_range=1)

        
            
        
        save_image(test_pred, './predict/test_pred{}_acc_{:.4f}_psnr_{:.4f}.png'.format(i+1, acc, psnr))