import os
import cv2
import numpy as np
import perspectTransform
import matplotlib.pyplot as plt
from featureMatch import getFeature_klt
from featureMatch import getFeature_tra
from featureMatch import getFeature_lisrd


pts = []
def draw_roi(event, x, y, flags, param):
    img2 = param[0].copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键添加
        pts.append((x, y))
        # pts.append([x, y])
        
    if event == cv2.EVENT_RBUTTONDOWN:
        # 鼠标右键删除
        pts.pop()

    if len(pts) > 0:
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)
        # cv2.circle(img2, tuple(pts[-1]), 3, (0, 0, 255), -1)

    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
    # TODO
    # TODO
    # if len(pts) > 3:
    #     for i in range(len(pts) - 1):
    #         cv2.circle(img2, tuple(pts[-1]), 5, (0, 0, 255), -1)
    #     cv2.polylines(img2, [np.array(pts)], True, (0, 255, 0), 2)

    cv2.imshow(param[1], img2)


if __name__ == "__main__":
    cap        = cv2.VideoCapture("./data/door.mp4")
    # match_img  = cv2.imread("./data/pu.png")
    img_insert = cv2.imread("./data/coke2.jpg")
    ret, zero_frame = cap.read()
    #--------------------------------------------#
    # 第 0 帧用于确定画图区域
    #--------------------------------------------#
    cv2.namedWindow("draw area of replace in first frame", 0)
    cv2.setMouseCallback("draw area of replace in first frame", \
                        draw_roi, \
                        param=[zero_frame, "draw area of replace in first frame"])
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("s"):
            break
    poly = np.float32(pts).reshape(-1, 1, 2)
    cv2.destroyAllWindows()
    h, w = img_insert.shape[:2]
    img_poly = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    msk_insert = np.zeros([h, w]).astype("float32")

    zero_frame_gray = cv2.cvtColor(zero_frame, cv2.COLOR_BGR2GRAY)
    x1, y1 = poly[0][0]
    x2, y2 = poly[2][0]
    match_img = zero_frame[int(y1): int(y2), int(x1): int(x2)]
    plt.imshow(match_img), plt.show()
    match_img_gray  = cv2.cvtColor(match_img, cv2.COLOR_BGR2GRAY)
    #--------------------------------------------#
    # 可以不gray
    # TODO
    #--------------------------------------------#
    feature_mode = 'lisrd'
    if feature_mode == "tra":
        _, p0 = getFeature_tra.getFeature_Tra(match_img_gray, zero_frame_gray, flag="surf")
    elif feature_mode == "klt":
        feature_params = dict(maxCorners=100,
                                qualityLevel=0.3,
                                minDistance=7,
                                blockSize=7)
        p0 = cv2.goodFeaturesToTrack(match_img_gray, mask=None, **feature_params)
    elif feature_mode == "lisrd":
        _, p0 = getFeature_lisrd.getFeature_Lisrd(match_img, zero_frame)
        
    old_frame = zero_frame
    old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    color = np.random.randint(0, 255, (1000, 3))
    mask = np.zeros_like(old_frame)
    while True:
        ret, cur_frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        
        #------------------------------------------------------#
        # 接入目标检测 缩小光流跟踪范围
        # TODO
        #------------------------------------------------------#
        cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        src_pts, dst_pts = getFeature_klt.getFeature_KLT(old_frame_gray, cur_frame_gray, p0)
        
        #------------------------------------------------------#
        # 开始 warp
        #------------------------------------------------------#
        M = perspectTransform.imgPerspectiveTransform(src_pts, dst_pts, poly, img_poly)

        img_insert_warped = cv2.warpPerspective(img_insert, M, 
                                                (cur_frame.shape[1], cur_frame.shape[0]), 
                                                borderMode=cv2.BORDER_CONSTANT, 
                                                borderValue=0)
        msk_insert_warped = cv2.warpPerspective(msk_insert, M, 
                                                (cur_frame.shape[1], cur_frame.shape[0]),
                                                borderMode=cv2.BORDER_CONSTANT, 
                                                borderValue=1)
        mask_warped_c3 = np.repeat(np.expand_dims(msk_insert_warped, 2), 3, axis=2)
        mask_warped = mask_warped_c3[: cur_frame.shape[0], : cur_frame.shape[1], :]
        frame = (mask_warped * cur_frame + img_insert_warped).astype('uint8')
        cv2.imshow('frame', frame)

        #-------------------------------------------------------#
        # 用于显示光流跟踪是否准确
        #-------------------------------------------------------#
        # for i, (old, new) in enumerate(zip(src_pts, dst_pts)):
        #     a, b = old.ravel()
        #     c, d = new.ravel()
        #     mask  = cv2.line(mask, (int(c), int(d)), (int(a), int(b)), color[i].tolist(), 2)
        #     frame = cv2.circle(cur_frame, (int(c), int(d)), 5, color[i].tolist(), -1)
        
        # img = cv2.add(frame, mask)
        # # cv2.namedWindow("optical flow", 1)
        # cv2.imshow('optical flow', img)
        
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
        old_frame_gray = cur_frame_gray.copy()
        p0 = dst_pts
        
    cv2.destroyAllWindows()
