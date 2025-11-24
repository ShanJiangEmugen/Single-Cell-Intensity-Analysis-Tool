import cv2
import sys
import numpy as np
from aicsimageio import AICSImage

# 记录上一个点击的坐标
prev_x, prev_y = -1, -1

def click_event(event, x, y, flags, params):
    global prev_x, prev_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if prev_x != -1 and prev_y != -1:
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            print(f"Distance to previous click: {distance:.2f} pixels")
        
        prev_x, prev_y = x, y

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"{x}, {y}", (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    if event == cv2.EVENT_RBUTTONDOWN:
        if prev_x != -1 and prev_y != -1:
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            print(f"Distance to previous click: {distance:.2f} pixels")

        prev_x, prev_y = x, y

        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, f"{b}, {g}, {r}", (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)

if __name__ == "__main__":
    image_fn = sys.argv[1]
    czi_img = AICSImage(image_fn)
    img = czi_img.get_image_data("YX", C=0, S=0, T=0)
    img = img.astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_DEEPGREEN)

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
