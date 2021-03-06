import cv2
import numpy as np
from matplotlib import pyplot as plt
LIST_GOOD_MATCHES = []
LIST_ACCURACY = []
img1 = None
win_name = 'Camera Matching'
MIN_MATCH = 10
# ORB Detector generation  ---①
detector = cv2.ORB_create(1000)
#------------------------------------------------------
###detector = cv2.BRISK_create()
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#--------------------------------------------------------
# note = cv2.imread('note4.png')
# h1, w1 = note.shape[:2]
# pts1=np.float32([[0,0],[w1,0],[0,h1],[w1,h1]])
# pts2 = None
# positions, positions2 = None, None
#--------------------------------------------------------

#////////////////////////////////////////////////////////done for multiple template matching
templates = []
#////////////////////////////////////////////////////////
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    if img1 is None:  # No image registered, camera bypass
        res = frame
    else:             # If there is a registered image, start matching
        img2 = frame
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #temp, offset = gray1.shape[:2]

        for tuple in templates:
            # Extract keypoints and descriptors
            img1 = tuple[2]
            kp1, desc1 = tuple[0], tuple[1]
            kp2, desc2 = detector.detectAndCompute(gray2, None)
            # k=2로 knnMatch
            matches = matcher.knnMatch(desc1, desc2, 2)
            # Good Match Point Extraction with 75% of Neighborhood Distance---②
            ratio = 0.75
            good_matches = [m[0] for m in matches \
                                if len(m) == 2 and m[0].distance < m[1].distance * ratio]
            ####print('good matches:%d/%d' %(len(good_matches),len(matches)))
            LIST_GOOD_MATCHES.append(len(good_matches))
            # Fill the mask with zeros to prevent drawing all matching points
            matchesMask = np.zeros(len(good_matches)).tolist()
            # if More than the minimum number of good matching points
            if len(good_matches) > MIN_MATCH:
                # Find coordinates of source and target images with good matching points ---③
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
                # Find Perspective Transformation Matrix ---⑤
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                accuracy=float(mask.sum()) / mask.size
                ####print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))
                LIST_ACCURACY.append(accuracy)
                if mask.sum() > MIN_MATCH:  # Set the mask to draw only outlier matching points if the normal
                    # number is more than the minimum number of matching points
                    matchesMask = mask.ravel().tolist()
                    # Area display after perspective conversion to original image coordinates  ---⑦
                    h,w, = img1.shape[:2]
                    pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
                    dst = cv2.perspectiveTransform(pts,mtrx)
                    #print(dst)
                    x1 = dst[0][0][0]
                    y1 = dst[0][0][1]
                    x2 = dst[1][0][0]
                    y2 = dst[1][0][1]
                    x4 = dst[2][0][0]
                    y4 = dst[2][0][1]
                    x3 = dst[3][0][0]
                    y3 = dst[3][0][1]

                    #print(dst)
                    #input('wait')

                    # pts2 = [[x1+offset, y1], [x3+offset, y3], [x2+offset, y2], [x4+offset, y4]]
                    # positions = pts2
                    # positions2 = [[x1+offset, y1], [x3+offset, y3], [x4+offset, y4], [x2+offset, y2]]


                    gray2 = cv2.polylines(gray2,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA) #poly lines
        # Draw match points with mask ---⑨
        ####res = cv2.drawMatches(gray1, kp1, gray2, kp2, good_matches, None, matchesMask=matchesMask, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        res = gray2
        #--------------------------------------------------------------------------
        #height, width = res.shape[:2]
        # pts2 = np.float32(pts2)
        # h,mask = cv2.findHomography(srcPoints=pts1,dstPoints=pts2,method=cv2.RANSAC, ransacReprojThreshold=5.0)
        # height, width, channels = res.shape
        # im1Reg = cv2.warpPerspective(note, h, (width, height))
        # mask2 = np.zeros(res.shape, dtype=np.uint8)
        # roi_corners2 = np.int32(positions2)
        # channel_count2 = res.shape[2]
        # ignore_mask_color2 = (255,) * channel_count2
        # cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)
        # mask2 = cv2.bitwise_not(mask2)
        # masked_image2 = cv2.bitwise_and(res, mask2)
        # res = cv2.bitwise_or(im1Reg, masked_image2)
        # --------------------------------------------------------------------------

    # Result output

    cv2.imshow(win_name, res)
    key = cv2.waitKey(1)
    if key == 27:    # Esc, 종료
            break
    elif key == ord(' '): # Set img1 by setting ROI to space bar
        x,y,w,h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            kp, desc = detector.detectAndCompute(gray1, None)
            tup = [kp,desc,img1]
            templates.append(tup)



else:
    print("can't open camera.")

cap.release()
cv2.destroyAllWindows()

AVERAGE_ACCURACY = sum(LIST_ACCURACY)/len(LIST_ACCURACY)
AVERAGE_MATCHES = sum(LIST_GOOD_MATCHES)/len(LIST_GOOD_MATCHES)
print(AVERAGE_ACCURACY)
print(AVERAGE_MATCHES)