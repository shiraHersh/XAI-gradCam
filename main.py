from torch import nn
from torchvision import models, transforms
from gradcam import GradCAM
from train_CNN import *
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import PIL

VISUALIZE_SIZE = (224, 224)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = models.resnet34(pretrained=False, num_classes=2)
# from ResNet34 import ResNet34
# cnn = ResNet34()
cnn = cnn.to(device)
params = {'batch_size': 1,
          'shuffle': False}

data = glob.glob(os.path.join('data', 'test', '*.jpg'))
list_IDs = list(map(lambda x: x[-12:-6], data))
labels = list(map(lambda x: int(x[-5:-4]), data))

channel_means = (0.5226, 0.4494, 0.4206)
channel_stds = (0.2411, 0.2299, 0.2262)

testTransform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(channel_means, channel_stds)
])

# Generators
testing_set = MaskDataset(list_IDs, labels, testTransform, train=False)
testing_generator = DataLoader(testing_set, **params)

cnn.load_state_dict(torch.load('model/latest.pkl', map_location=lambda storage, loc: storage))
# cnn.train()

grad_cam = GradCAM(model=cnn, feature_layer=list(cnn.layer3.modules())[-1])

predictions = []
y_true = []
for idx, (images, labels) in enumerate(testing_generator):
    images = images.to(device)
    # outputs = cnn(images).view(-1).detach().cpu()

    model_output = grad_cam.forward(images)
    target = model_output.argmax(1).item()

    grad_cam.backward_on_target(model_output, target)

    # Get feature gradient
    feature_grad = grad_cam.feature_grad.data.cpu().numpy()[0]
    # Get weights from gradient
    weights = np.mean(feature_grad, axis=(1, 2))  # Take averages for each gradient
    # Get features outputs
    feature_map = grad_cam.feature_map.data.cpu().numpy()
    # grad_cam.clear_hook()

    # Get cam
    cam = np.sum((weights * feature_map.T), axis=2).T
    cam = np.maximum(cam, 0)  # apply ReLU to cam


    cam = cv2.resize(cam, VISUALIZE_SIZE)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = 1-cam

    image = PIL.Image.open(data[idx])
    image_orig_size = image.size
    org_img = np.asarray(image.resize(VISUALIZE_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img_with_heatmap = heatmap + np.float32(org_img)/255
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    org_img = cv2.resize(org_img, image_orig_size)

    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.imshow(org_img)
    plt.subplot(1,2,2)
    plt.imshow(cv2.resize(np.uint8(255 * img_with_heatmap), image_orig_size))
    plt.savefig('images/{}_{}.jpg'.format(idx, labels[0]))
