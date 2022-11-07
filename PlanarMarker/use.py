import os
import cv2
import numpy as np
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

    cv2.imshow(param[1], img2)

if __name__ == "__main__":
    #------------------------------------------------#
    # 设置参数
    #------------------------------------------------#
    save_video           = True
    show_flow            = False
    interact             = False
    feature_extract_mode = 'lisrd'

    video_path = "./data/door.mp4"
    cap = cv2.VideoCapture(video_path)
    img_insert_path = "./data/coke2.jpg"
    img_insert_bgr  = cv2.imread(img_insert_path)
    
    ret, zero_frame = cap.read()
    #------------------------------------------------#
    # 第 0 帧用于 画代替 区域 | 用于匹配
    #------------------------------------------------#
    if interact:
        cv2.namedWindow("draw", 0)
        cv2.setMouseCallback("draw", draw_roi, param=[zero_frame, "draw"])
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("s"):
                break
        poly = np.float32(pts).reshape(-1, 1, 2)
        #--------------------------------------------#
        # TODO 截图多边形 | 矫正
        #--------------------------------------------#
        x1, y1 = poly[0][0]
        x2, y2 = poly[2][0]
        area_replaced = zero_frame[int(y1): int(y2), int(x1): int(x2)]
        print("显示选择的区域............")
        cv2.imshow('area_replaced', area_replaced)
        cv2.waitKey(4000)
        cv2.destroyAllWindows()
    else:
        matched_img_path  = "./data/j.jpg"
        matched_img = cv2.imread(matched_img_path)
        area_replaced = matched_img
        h, w = matched_img.shape[:2]
        poly = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    #-----------------------------------------------#
    # 用 第0帧瞄准的区域 和 第0帧做特诊匹配
    #-----------------------------------------------#
    if feature_extract_mode == 'lisrd': # (64, 1, 2)
        print("first frame used lisrd...")
        _, p0 = getFeature_lisrd.getFeature_Lisrd(area_replaced, zero_frame)
    elif feature_extract_mode == 'klt':
        #-------------------------------------------#
        # 提取到的特征点不足以求得homography
        #-------------------------------------------#
        print("first frame used klt...")
        feature_params = dict(maxCorners=100,
                                qualityLevel=0.3,
                                minDistance=7,
                                blockSize=7)
        area_replaced = cv2.cvtColor(area_replaced, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(area_replaced, mask=None, **feature_params)
    else: # (170, 1, 2)
        print("first frame used surf...")
        _, p0 = getFeature_tra.getFeature_Tra(area_replaced, zero_frame)
    
    
    origin_poly = poly
    img_insert_rgb  = img_insert_bgr
    h, w = img_insert_rgb.shape[:2]
    img_insert_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    img_insert_msk = np.zeros([h, w]).astype('float32')
    
    #-----------------------------------------------#
    # 保存视频路径
    #-----------------------------------------------#
    if save_video:
        save_video_path = './data/{}_tracking_{}.mp4'.format(\
                                    os.path.basename(video_path).split('.')[0], 
                                    str(feature_extract_mode))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (zero_frame.shape[1], zero_frame.shape[0]) 
        video_writer = cv2.VideoWriter(save_video_path, fourcc, 30, frame_size)
        video_writer.write(zero_frame)

    old_frame = zero_frame
    color = np.random.randint(0, 255, (1000, 3))
    mask = np.zeros_like(zero_frame)
    zero_frame_gray = cv2.cvtColor(zero_frame, cv2.COLOR_BGR2GRAY)
    # tmp_frame = zero_frame

    while True:
        ret, cur_frame = cap.read()

        x1, y1 = origin_poly[0][0]
        x2, y2 = origin_poly[2][0]
        area_replaced = old_frame[int(y1): int(y2), int(x1): int(x2)]
        if ret:
            # old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            good_old, good_new = getFeature_klt.getFeature_KLT(zero_frame_gray, cur_frame_gray, p0)
            # good_old, good_new = getFeature.getFeature_KLT(old_frame_gray, cur_frame_gray, p0)
            #--------------------------------------------------#
            # 跟踪到的点少于4个则用传统方法，从新匹配
            # 跟踪丢了，使用上一帧的结果
            #--------------------------------------------------#
            if len(good_new) < 10:
                print("rematch feature points")
                #--------------TODO----------------------------#
                # 如何解决丢失问题
                # 似乎可以删除没有移动的点
                #----------------------------------------------#
                _, p0 = getFeature_klt.getFeature(area_replaced, cur_frame)
                good_old, good_new = getFeature_klt.getFeature_KLT(zero_frame_gray, cur_frame_gray, p0)

            src_pts, dst_pts = good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2)
            print("count feature keypoints {}".format(len(dst_pts)))

            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            pts = np.float32(origin_poly).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            dst = np.float32(dst).reshape(-1, 1, 2)
            #-------------------------------------------------------#
            # 用于显示光流跟踪是否准确
            #-------------------------------------------------------#
            if show_flow:
                for i, (old, new) in enumerate(zip(src_pts, dst_pts)):
                    a, b = old.ravel()
                    c, d = new.ravel()
                    mask  = cv2.line(mask, (int(c), int(d)), (int(a), int(b)), color[i].tolist(), 2)
                    frame = cv2.circle(cur_frame, (int(c), int(d)), 5, color[i].tolist(), -1)
                
                frame = cv2.add(frame, mask)
                # cv2.namedWindow("optical flow", 1)
                cv2.imshow('optical flow', frame)

            else:
                M = cv2.getPerspectiveTransform(img_insert_pts, dst)
                img_insert_warped = cv2.warpPerspective(img_insert_rgb, M, 
                                                        (cur_frame.shape[1], cur_frame.shape[0]), 
                                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                msk_insert_warped = cv2.warpPerspective(img_insert_msk, M, 
                                                        (cur_frame.shape[1], cur_frame.shape[0]),
                                                        borderMode=cv2.BORDER_CONSTANT, borderValue=1)
                mask_warped_c3 = np.repeat(np.expand_dims(msk_insert_warped, 2), 3, axis=2)
                mask_warped = mask_warped_c3[: cur_frame.shape[0], : cur_frame.shape[1], :]
                frame = (mask_warped * cur_frame + img_insert_warped).astype('uint8')
                cv2.imshow('frame', frame)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            if save_video:
                video_writer.write(frame)
            zero_frame_gray = cur_frame_gray.copy()
            # old_frame = cur_frame.copy()
            p0 = good_new.reshape(-1, 1, 2)
            origin_poly = dst
        else:
            print("end")
            break
    if save_video:
        video_writer.release()
    cv2.destroyAllWindows()