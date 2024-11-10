import cv2
import cv2 as cv
import numpy as np

fast = cv2.SIFT_create()

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    if h > sh or w > sw:
        interp = cv.INTER_AREA
    else:
        interp = cv.INTER_CUBIC

    aspect = w/h

    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor]*3

    scaled_img = cv.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv.BORDER_CONSTANT, value=padColor)

    return scaled_img

def KetchupDetection(fridge1ImageString):
    image = cv2.imread('Ketchup.png')
    image2 = fridge1ImageString

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    kp, des1 = fast.detectAndCompute(image, None)
    kp2, des2 = fast.detectAndCompute(image2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    isKetchup = False;
    if len(good) > 30:
        isKetchup = True

    if isKetchup:
        img3 = cv.drawMatchesKnn(image, kp, image2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        img3 = image2
    return good, img3, isKetchup

def BBQDetection(fridge2imagestring):
    image = cv2.imread('BBQ.png')
    image2 = fridge2imagestring

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    kp, des1 = fast.detectAndCompute(image, None)
    kp2, des2 = fast.detectAndCompute(image2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    isBBQ = False
    if len(good) > 30:
        isBBQ = True
    if isBBQ:
        img3 = cv.drawMatchesKnn(image, kp, image2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        img3 = image2
    return good, img3, isBBQ

def Siracha(fridge2imagestring):
    image = cv2.imread('Siracha.png')
    image2 = fridge2imagestring

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    lowerOrange = np.array([40,100,100])
    upperOrange = np.array([80, 255, 255])

    mask1 = cv2.inRange(image, lowerOrange, upperOrange)
    mask2 = cv2.inRange(image2, lowerOrange, upperOrange)

    kernel = np.ones((40, 10), np.uint8)
    dilated_mask = cv.dilate(mask1, kernel, iterations=1)
    dilated_mask2 = cv.dilate(mask2, kernel, iterations=1)

    masked_img1 = cv.bitwise_and(image, image, mask=dilated_mask)
    masked_img2 = cv.bitwise_and(image2, image2, mask=dilated_mask2)

    # Find contours on the dilated mask
    # contours, _ = cv.findContours(dilated_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours2, _2 = cv.findContours(dilated_mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #
    # contoursSorted = sorted(contours, key= lambda x: cv2.contourArea(x))
    # contoursSorted2 = sorted(contours2, key=lambda x: cv2.contourArea(x))
    #
    # area1 = cv2.contourArea(contours[len(contours) - 1])
    # area2 = cv2.contourArea(contours2[len(contours2) - 1])
    #
    # # Draw merged contours for visualization
    # output_img = image.copy()
    # output_img2 = image2.copy()
    # cv.drawContours(output_img, [contoursSorted[len(contours) - 1]], -1, (255, 0, 0), 3)
    # cv.drawContours(output_img2, [contoursSorted2[len(contours2) - 1]], -1, (255, 0, 0), 3)

    mean, std_dev = cv2.meanStdDev(masked_img2)

    isSiracha = False

    if np.all(std_dev < 1e-6):
        img3 = image2
        good = []
    else:
        fast = cv2.SIFT_create()

        kp, des1 = fast.detectAndCompute(masked_img1, None)
        kp2, des2 = fast.detectAndCompute(masked_img2, None)

        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        if len(good) > 10:
            isSiracha = True

        img3 = cv.drawMatchesKnn(image, kp, image2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return good, img3, isSiracha


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break


    good_BBQ, img_BBQ, isBBQ = BBQDetection(frame)
    good_ketchup, img_ketchup, isKetchup = KetchupDetection(frame)
    good_siracha, img_siracha, isSiracha = Siracha(frame)

    cv2.imshow('Ketchup Detection', img_ketchup)
    cv2.imshow('BBQ Detection', img_BBQ)
    cv2.imshow('Siracha Detection', img_siracha)

    if isKetchup:
        print("Ketchup Detected")
    elif isBBQ:
        print("BBQ Detected")
    elif isSiracha:
        print("Siracha Detected")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()