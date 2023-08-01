
from thop import profile
from unet import *
from ptflops import get_model_complexity_info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3,n_classes=3,bilinear=True).to(device)
input = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(input, ))
print('MACs = ' + str(macs/1000**3) + ' G')
print('FLOPs = ' + str(2*macs/1000**3) + ' G')
print('Params = ' + str(params/1000**2) + ' M')

with torch.cuda.device(0):
  net = UNet(n_channels=3,n_classes=3,bilinear=True).to(device)
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))