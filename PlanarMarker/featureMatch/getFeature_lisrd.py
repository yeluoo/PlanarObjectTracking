import cv2
import os
import numpy as np
import torch
import torch.nn.functional as func
import matplotlib.pyplot as plt

import sys
# sys.path.append("./")
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("D://pot//LISRD//")
from lisrd.models import get_model
from lisrd.models.base_model import Mode
from lisrd.models.keypoint_detectors import SP_detect, load_SP_net
from lisrd.utils.geometry_utils import extract_descriptors, lisrd_matcher, filter_outliers_ransac
# from utils import plot_images, plot_keypoints, plot_matches

def getFeature_Lisrd(img1, img2):
    img_size1 = img1.shape[:2]
    img_size2 = img2.shape[:2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the LISRD model
    model_config = {'name': 'lisrd', 'desc_size': 128, 'tile': 3, 'n_clusters': 8, 'meta_desc_dim': 128,
                    'learning_rate': 0.001, 'compute_meta_desc': True, 'freeze_local_desc': False}
    lisrd_net = get_model('lisrd')(None, model_config, device)
    checkpoint_path = 'D://pot//PlanarMarker//featureMatch//lisrd//weights//lisrd_vidit.pth'
    lisrd_net.load(checkpoint_path, Mode.EXPORT)
    lisrd_net._net.eval()

    # Load the keypoint model, here SuperPoint
    kp_net = load_SP_net(conf_thresh=0.015, cuda=torch.cuda.is_available(), nms_dist=4, nn_thresh=0.7)

    with torch.no_grad():
    # Keypoint detection
        kp1 = SP_detect(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), kp_net)
        gpu_kp1 = torch.tensor(kp1, dtype=torch.float, device=device)[:, :2]
        kp2 = SP_detect(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), kp_net)
        gpu_kp2 = torch.tensor(kp2, dtype=torch.float, device=device)[:, :2]
        
        # Descriptor inference
        gpu_img1 = torch.tensor(img1, dtype=torch.float, device=device)
        inputs1 = {'image0': gpu_img1.unsqueeze(0).permute(0, 3, 1, 2)}
        outputs1 = lisrd_net._forward(inputs1, Mode.EXPORT, model_config)
        desc1 = outputs1['descriptors']
        meta_desc1 = outputs1['meta_descriptors']
        gpu_img2 = torch.tensor(img2, dtype=torch.float, device=device)
        inputs2 = {'image0': gpu_img2.unsqueeze(0).permute(0, 3, 1, 2)}
        outputs2 = lisrd_net._forward(inputs2, Mode.EXPORT, model_config)
        desc2 = outputs2['descriptors']
        meta_desc2 = outputs2['meta_descriptors']
        
        # Sample the descriptors at the keypoint positions
        desc1, meta_desc1 = extract_descriptors(gpu_kp1, desc1, meta_desc1, img_size1)
        desc2, meta_desc2 = extract_descriptors(gpu_kp2, desc2, meta_desc2, img_size2)
        
        # Nearest neighbor matching based on the LISRD descriptors
        matches = lisrd_matcher(desc1, desc2, meta_desc1, meta_desc2).cpu().numpy()
    matched_kp1, matched_kp2 = kp1[matches[:, 0]][:, [1, 0]], kp2[matches[:, 1]][:, [1, 0]]
    matched_kp1, matched_kp2 = filter_outliers_ransac(matched_kp1, matched_kp2)

    src_pts = np.float32(matched_kp1).reshape(-1, 1, 2)
    dst_pts = np.float32(matched_kp2).reshape(-1, 1, 2)

    return src_pts, dst_pts
