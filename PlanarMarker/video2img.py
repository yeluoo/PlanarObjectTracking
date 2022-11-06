import os
import cv2
import time

def video2img(path, save_path):
    cap = cv2.VideoCapture(path)

    frameToStart = 0  # 开始帧 = 开始时间*帧率
    frametoStop = 100  # 结束帧 = 结束时间*帧率
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)  # 设置读取的位置,从第几帧开始读取视频

    current_frame = frameToStart
    while (1):
        success, frame = cap.read()
        if success:
            current_frame += 1
            if (current_frame >= 0 and current_frame <= frametoStop):
                cv2.imwrite(os.path.join(save_path, str(current_frame) + '.png'), frame)
            else:
                break
        else:
            print('end')
            break

def video2clip(path, save_path):
    start=time.time()
    cap = cv2.VideoCapture(path)

    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print('视频总帧数：{} \t 视频帧速：{} \t 视频大小：{}，{}'.format(total, fps, h, w))

    size = (int(w), int(h))  # 原视频的大小
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, size)

    frameToStart = int(1690 * fps)  # 开始帧 = 开始时间*帧率
    frametoStop = int(1770 * fps)  # 结束帧 = 结束时间*帧率
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)  # 设置读取的位置,从第几帧开始读取视频

    current_frame = frameToStart
    while (1):
        success, frame = cap.read()
        if success:
            current_frame += 1
            if (current_frame >= 0 and current_frame <= frametoStop):
                videoWriter.write(frame)
            else:
                break
        else:
            print('end')
            break
    end=time.time()
    print(end-start)

path = "./data/door.mp4"
save_path = "./data/video2img/"
os.makedirs(save_path, exist_ok=True)
video2img(path, save_path)
