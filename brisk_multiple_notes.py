import cv2
import numpy as np
from PIL import Image, ImageDraw
from math import ceil

templates = []
def createNote(text):
    width = 200
    height = len(text)/28
    height = ceil(height)*10+50
    img = Image.new('RGB', (width, height), color=(0,0,0))
    d = ImageDraw.Draw(img)
    part = ''
    line = 10
    for i in range(1,len(text)+1):
        if i%28 == 0:
            d.text((10, line), part, fill=(255, 255, 255))
            part = ''
            line += 10
        else:
            part += text[i-1]
    d.text((10, line), part, fill=(255, 255, 255))

    img.save('TEXT-TO-IMAGE/'+str(len(templates)+1)+'.png')


img1 = None
win_name = 'Camera Matching'
MIN_MATCH = 10 # previous = 10
# ORB Detector generation  ---①
#detector = cv2.ORB_create(nfeatures=3000)
#------------------------------------------------------
detector = cv2.BRISK_create()
#------------------------------------------------------
# Flann Create extractor ---②
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
# Camera capture connection and frame size reduction ---③

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#--------------------------------------------------------
pts1 = None
pts2 = None
positions, positions2 = None, None
#--------------------------------------------------------
res = None
while cap.isOpened():
    # frames += 1
    ret, frame = cap.read()
    if ret == False:
        break
    if img1 is None:  # No image registered, camera bypass
        res = frame
    else:             # If there is a registered image, start matching
        img2 = frame
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(gray2, None)  # keypoints and descriptors of current video frame

        for tuple in templates:
            # Extract keypoints and descriptors
            img1 = tuple[2]
            kp1, desc1 = tuple[0], tuple[1]
            #=======================================================
            note = tuple[3] # Note Image for current reference Image
            h1, w1 = note.shape[:2]
            pts1=np.float32([[0,0],[w1,0],[0,h1],[w1,h1]]) # coordinates of Note Image
            pts2 = None
            positions, positions2 = None, None

            # k=2 knnMatch
            matches = matcher.knnMatch(desc1, desc2, 2)
            # Good Match Point Extraction with 75% of Neighborhood Distance---②
            ratio = 0.75
            good_matches = [m[0] for m in matches \
                                if len(m) == 2 and m[0].distance < m[1].distance * ratio]
            # Fill the mask with zeros to prevent drawing all matching points
            matchesMask = np.zeros(len(good_matches)).tolist()
            # if More than the minimum number of good matching points
            if len(good_matches) > MIN_MATCH:
                # Find coordinates of source and target images with good matching points ---③
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
                # Find Perspective Transformation Matrix ---⑤
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask.sum() > MIN_MATCH:  # Set the mask to draw only outlier matching points if the normal
                    # number is more than the minimum number of matching points
                    matchesMask = mask.ravel().tolist()
                    # Area display after perspective conversion to original image coordinates  ---⑦
                    h,w, = img1.shape[:2]
                    pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
                    dst = cv2.perspectiveTransform(pts,mtrx)
                    x1 = dst[0][0][0]
                    y1 = dst[0][0][1]
                    x2 = dst[1][0][0]
                    y2 = dst[1][0][1]
                    x4 = dst[2][0][0]
                    y4 = dst[2][0][1]
                    x3 = dst[3][0][0]
                    y3 = dst[3][0][1]

                    offset = 0
                    pts2 = [[x1+offset, y1], [x3+offset, y3], [x2+offset, y2], [x4+offset, y4]]
                    positions = pts2
                    positions2 = [[x1+offset, y1], [x3+offset, y3], [x4+offset, y4], [x2+offset, y2]]


                    cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA) #poly lines
                    res = img2
                    #--------------------------------------------------------------------------
                    height, width = res.shape[:2]
                    pts2 = np.float32(pts2)
                    h,mask = cv2.findHomography(srcPoints=pts1,dstPoints=pts2,method=cv2.RANSAC, ransacReprojThreshold=5.0)
                    height, width, channels = res.shape
                    im1Reg = cv2.warpPerspective(note, h, (width, height))
                    mask2 = np.zeros(res.shape, dtype=np.uint8)
                    roi_corners2 = np.int32(positions2)
                    channel_count2 = res.shape[2]
                    ignore_mask_color2 = (255,) * channel_count2
                    cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)
                    mask2 = cv2.bitwise_not(mask2)
                    masked_image2 = cv2.bitwise_and(res, mask2)
                    res = cv2.bitwise_or(im1Reg, masked_image2)
                    img2 = res
        res = img2
    cv2.imshow(win_name, res)
    key = cv2.waitKey(1)
    if key == 27:    # Esc
            break
    elif key == ord(' '): # Set img1 by setting ROI to space bar
        note = input('Enter your Note: ')
        #print(note)
        note = 'hi this is a test note....'
        x,y,w,h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            kp, desc = detector.detectAndCompute(gray1, None)
            tup = [kp,desc,img1]
            createNote(note)
            print(note)
            noteImage = cv2.imread('TEXT-TO-IMAGE/'+str(len(templates)+1)+'.png')
            tup.append(noteImage)
            templates.append(tup)

else:
    print("can't open camera.")

cap.release()
cv2.destroyAllWindows()
