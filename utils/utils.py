import cv2
import numpy as np


def get_slices(image: np.ndarray, slice_width: int, slice_height: int, overlapped_ratio: float) -> list[np.ndarray]:
    window = []
    h, w = image.shape[:2]
    stepSizeY = int(slice_height - slice_height*overlapped_ratio)
    stepSizeX = int(slice_width - slice_width*overlapped_ratio)
    for y in range(0, image.shape[0], stepSizeY):
        for x in range(0, image.shape[1], stepSizeX):
            if w-x < stepSizeX and h-y < stepSizeY:
                window.append(((h-slice_height, w-slice_width), image[h-slice_height:h, w-slice_width:w, :]))
            elif w-x < stepSizeX:
                  window.append(((y, w-slice_width), image[y:y + slice_width, w-slice_width:w, :]))
            elif h-y < stepSizeY:
                  window.append(((h-slice_height, x), image[h-slice_height:h, x:x + slice_height,:]))
            else:
                  window.append(((y, x), image[y:y + slice_width, x:x + slice_height,:]))
    return window

def nmm(bboxes: np.ndarray, iou_thres: float):
    ...

def draw_bbox(image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    img = image
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.5
    for i in range(bboxes.shape[0]):   
        x1 = int(bboxes[i,0])
        y1 = int(bboxes[i,1])
        w = int(bboxes[i,2])
        h = int(bboxes[i,3])
        text =  str(int(bboxes[i,4])) + ' ' + str(bboxes[i,5])
        labelSize = cv2.getTextSize(text, fontFace, fontScale, 1)
        start_point = (x1, y1)
        end_point = (x1 + w, y1 + h)
        img = cv2.rectangle(img, start_point, end_point,(255,0,0), 2)
        labelSize = cv2.getTextSize(text, fontFace, fontScale, 1)
        img = cv2.rectangle(img, (x1, y1 - labelSize[0][1]), (x1+labelSize[0][0], y1), (60,60,60), cv2.FILLED)
        img = cv2.putText(img, text, start_point, fontFace, fontScale, (0,255,0), thickness=1)
    return img
