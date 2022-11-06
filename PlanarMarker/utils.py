import os
import cv2
import glob

# 追踪鼠标
pts = []
def draw_roi(event, x, y, flags, param):
    img2 = param[0].copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))

    if event == cv2.EVENT_RBUTTONDOWN:
        pts.pop()

    if len(pts) > 0:
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv2.imshow(param[1], img2)


def get_frames(video_name):
    #--------------------------------------------------#
    # 支持 3 种方式 1. 调用摄像头 2. 本地视频 3. 本地图片集
    #--------------------------------------------------#
    count = 0
    frames = {}
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                frames[count] = frame
                count += 1
                yield frames
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('.mp4') or \
            video_name.endswith('.MP4') or \
            video_name.endswith('.mov'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                # cv2.imshow('frame', frame)
                # cv2.waitKey(40)
                # cv2.destroyAllWindows()
                frames[count] = frame
                count += 1
                yield frames
            else:
                break
    else:
        images = glob.glob(os.path.join(video_name, '*.[jp*][pn]g'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            frames[count] = frames
            count += 1
            yield frame

if __name__ == "__main__":
    
    # get_frames("./data/door.mp4")
    video_name = "./data/door.mp4"
    video_name = "./data/video2img/"
    for i, j in enumerate(get_frames(video_name)):
        print("第.{}.张图片".format(i))
        cv2.imshow(str(i), j[i])
        cv2.waitKey(40)
        break
        # cv2.destroyAllWindows()