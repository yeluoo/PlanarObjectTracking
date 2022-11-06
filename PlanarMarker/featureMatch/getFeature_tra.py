import os
import cv2
import numpy as np

def getFeature_Tra(ref_frame, 
                    det_frame, 
                    flag='sift', 
                    match_method="KNN"):
    """
    input : numpy gray
    output: coordinate
    """
    #-------------------------------------------------#
    # 定义传统特征提取的类别
    #-------------------------------------------------#
    if flag =='sift':
        sift = cv2.xfeatures2d.SIFT_create() #3.4.2.16
    elif flag == "surf":
        sift = cv2.xfeatures2d.SURF_create()
    elif flag == "orb":     # 使用orb match_method = BF
        sift = cv2.ORB_create()
    elif flag == "fast":
        # TODO
        pass
    #------------------------------------------------#
    # 寻找特征点与描述子
    #------------------------------------------------#
    kp1, des1 = sift.detectAndCompute(ref_frame, None)
    kp2, des2 = sift.detectAndCompute(det_frame, None)
    #-------------------------------------------------#
    # 寻找最佳匹配的方法
    # 支持 KNN / BF
    if match_method == "KNN":
        #---------------------------------------------#
        # 配套使用的参数
        # KnnMatch 
        # 返回描述子ref下标/det下标/特征点之间的距离
        #---------------------------------------------#
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        #--------------------------------------------#
        # 显示是否匹配结果
        # if ref_frame == rgb
        #--------------------------------------------#
        # img = cv2.drawMatches(ref_frame, kp1, det_frame, kp2, good, None, flags=2)
        # cv2.imshow("matches", img)
        # cv2.waitKey(4000)
        # cv2.destroyAllWindows()
        
    elif match_method == "BF":
        bf = cv2.BFMatcher_create()
        matches = bf.match(des1, des2)
        good = matches
    #-------------------------------------------------#
    # 匹配点数超过 10 个才算有效
    #-------------------------------------------------#
    MIN_MATCH_COUNT = 10
    assert len(good) > MIN_MATCH_COUNT
    #-------------------------------------------------#
    # kp每个特征点本身具有 4 个特性
    # pt:关键点坐标，angle：表示关键点方向，
    # response表示响应强度，size:该点的直径大小
    #-------------------------------------------------#
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

    return src_pts, dst_pts

if __name__ == "__main__":
    im1 = cv2.imread('./data/video2img/1.png')
    im2 = cv2.imread('./data/video2img/2.png')
    print(im1, im2)
    img1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    color = np.random.randint(0, 255, (1000, 3))
    src_pts, dst_pts = getFeature_Tra(img1_gray, img2_gray)
    mask = np.hstack([im1, im2])
   
    for i, (old, new) in enumerate(zip(src_pts, dst_pts)):
        a, b = old.ravel()
        c, d = new.ravel()
        mask  = cv2.line(mask, (int(c) + im2.shape[1], int(d)), (int(a), int(b)), color[i].tolist(), 2)

    cv2.imshow('frame', mask)
    cv2.waitKey(30000) 