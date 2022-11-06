import os
import cv2
import numpy as np

def getFeature_KLT(old_frame, cur_frame, p0):
    """
    input: numpy gray
    ouput: coordinate
    """
    # old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    # cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_RGB2GRAY)
    #---------------------------------------------#
    # 光流法参数
    # maxLevel 未使用的图像金字塔层数
    #---------------------------------------------#
    lk_params = dict(winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #---------------------------------------------#
    # 参考帧 | 检测帧 | 参考帧特征点向量
    #---------------------------------------------#
    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, cur_frame_gray, p0, None, **lk_params)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_frame, cur_frame, p0, None, **lk_params)
    #---------------------------------------------#
    # 选择good points
    # 找到的特征点少于10个则用传统算法找特征点
    #---------------------------------------------#
    if len(p1[st == 1]) <= 10:  
        # TODO
        assert len(p1[st == 1]) <= 10
    print("tracking -{}- featrue...".format(len(p1[st == 1])))

    if p1 is not None:
        good_old = p0[st == 1]
        good_new = p1[st == 1]

    src_pts = np.float32(good_old).reshape(-1, 1, 2)
    dst_pts = np.float32(good_new).reshape(-1, 1, 2)

    return src_pts, dst_pts
    


if __name__ == "__main__":

    cap = cv2.VideoCapture("./data/door.mp4")
    match_img  = cv2.imread("./data/pu.png")
    feature_params = dict(maxCorners=100,
                                qualityLevel=0.3,
                                minDistance=7,
                                blockSize=7)
    ret, zero_frame = cap.read()
    zero_frame_gray = cv2.cvtColor(zero_frame, cv2.COLOR_BGR2GRAY)
    match_img_gray = cv2.cvtColor(match_img, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(zero_frame_gray, mask=None, **feature_params)
    # p0 = cv2.goodFeaturesToTrack(match_img_gray, mask=None, **feature_params)
    color = np.random.randint(0, 255, (1000, 3))

    old_frame = zero_frame
    mask = np.zeros_like(old_frame)
    old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    while True:
        ret, cur_frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        
        cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        src_pts, dst_pts = getFeature_KLT(old_frame_gray, cur_frame_gray, p0)
        
        for i, (old, new) in enumerate(zip(src_pts, dst_pts)):
            a, b = old.ravel()
            c, d = new.ravel()
            mask  = cv2.line(mask, (int(c), int(d)), (int(a), int(b)), color[i].tolist(), 2)
            frame = cv2.circle(cur_frame, (int(c), int(d)), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
        old_frame_gray = cur_frame_gray.copy()
        p0 = dst_pts
        
    cv2.destroyAllWindows()