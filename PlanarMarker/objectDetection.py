import os
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.io import read_image
import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# 定义模型
model = models.detection.ssd300_vgg16(pretrained=True)
# img= cv2.imread('./data/video2img/33.png')
img = Image.open('./data/video2img/33.png')
img2 = read_image('./data/video2img/33.png')


# 数据预处理
transforms = transforms.Compose([
            # transfroms.Resize((416, 416))
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.255]
            )
])


img_trans = transforms(img).unsqueeze(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to("cpu")
img_trans.to("cpu")



model.eval()
with torch.no_grad():
    output = model(img_trans)
    
    # output = output.numpy().squeeze()
    # output = np.transpose(output, (1, 2, 0))
    # output = Image.fromarray(output)

    score_threshold = 0.85
    output = output[0]
    boxes=output['boxes'][output['scores'] > score_threshold]
    print(boxes)
    colors = ["blue", "yellow"]

    out_img = torchvision.utils.draw_bounding_boxes(img2, boxes, colors=colors, width=5)
    out_img=out_img.permute(1,2,0)
    plt.imshow(out_img)
    plt.show()
    

