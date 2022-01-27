import os
import cv2
import numpy as np
import imutils
import math

img_dir = 'images'
names = os.listdir(img_dir)
temp_image=[]

for i in range(math.ceil(len(names)/5)):
    images=[]
    if i !=math.ceil(len(names)/5)-1:
        for name in names[5*i:5*i+5]:
            # print(name)
            img_path = os.path.join(img_dir, name)
            image = cv2.imread(img_path)
            images.append(image)
    else:
        for name in names[5*i:]:
            # print(name)
            img_path = os.path.join(img_dir, name)
            image = cv2.imread(img_path)
            images.append(image)

    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)

    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    _, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    mask = np.zeros(thresh.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    minRect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    _, cnts, hierarchy = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    # stitched = stitched[y:y + h, x:x + w]
    temp_image.append(stitched)
# print(len(temp_image))
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
status, stitched = stitcher.stitch(temp_image)
stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
cv2.imwrite('generated.jpg', stitched)