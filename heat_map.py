from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2
import os
from unet import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_best = UNet(n_channels=3,n_classes=3,bilinear=True).to(device)
model_best.load_state_dict(torch.load("./models/UNet_best_at_epoch_55.ckpt"))
model_best.eval()

for name, child in model_best._modules.items():
    print(name)
target_layers = [model_best.inc]

image_path = 'testlabel/testlabel/004038.png'
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] #1是讀取RGB
rgb_img = np.float32(rgb_img) / 255
rgb_img = np.resize(rgb_img, (rgb_img.size()[0]// 2, rgb_img.size()[1]// 2))
#rgb_img = rgb_img.numpy()
#preprocess_image作用:正規化圖片，並轉成tensor
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

"""
testdata = ImageFolder('./test/', transform=transforms.ToTensor())
test_dataset = FishDataset(testdata)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
"""
#初始化CAM對象，包括模型，目標層以及是否使用cuda等
cam = GradCAM(model=model_best, target_layers=target_layers, use_cuda=True)
#選定目標類別，如果不設置，則默認為分數最高的那一類
# We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
targets = None
##for input_tensor in test_loader:
#計算CAM
# You can also pass aug_smooth = True and eigen_smooth = True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category = targets)
#展示heat map並保存
# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('Grad_CAM_cam.jpg', cam_image)


gb_model = GuidedBackpropReLUModel(model=model_best, use_cuda=True)
gb = gb_model(input_tensor, target_category=None)

cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
cam_gb = deprocess_image(cam_mask * gb)
gb = deprocess_image(gb)
cv2.imwrite('Grad_CAM_gb.jpg', gb)
cv2.imwrite('Grad_CAM_cam_gb.jpg', cam_gb)