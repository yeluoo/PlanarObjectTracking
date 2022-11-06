import os
import cv2
import numpy as np


def imgPerspectiveTransform(src_pts, dst_pts, input_poly, img_poly):
    """
    input : (参考帧 | 检测帧) 匹配 keypoints | poly
    output: 变换矩阵 3 * 3
    """
    #-----------------------------------------------------#
    # cv2.findHomography: 计算最优的变换矩阵 3 * 3
    # src_pts / dst_pts
    # 第三个参数 0 - 最小二乘  RANSAC - 基于RANSAC LMEDS - 最小中值 RHO - 基于PROSAC
    # 第四个参数 误差阈值 取值在1-10
    # 返回M是单应矩阵  mask是在线的点
    #-----------------------------------------------------#
    #-----------------------------------------------------#
    # 两张图的变换矩阵
    #-----------------------------------------------------#
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #-----------------------------------------------------#
    # 参考帧的coordinate --> 检测帧的coordinate
    #-----------------------------------------------------#
    tgt_ploy = cv2.perspectiveTransform(input_poly, H)
    #-----------------------------------------------------#
    # (img_insert的coordinate | 检测帧的ploy) 的 变换矩阵
    M = cv2.getPerspectiveTransform(img_poly, tgt_ploy)
    return M


if __name__  == "__main__":

    im1 = cv2.imread('./dataideo2img/11.png')
    im2 = cv2.imread('./dataideo2img/21.png')



