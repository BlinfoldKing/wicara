import numpy as np
import cv2
import imutils
from sklearn.metrics import pairwise

bg = None

def run_avg(image, aWeight):
    global bg

    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype('uint8'), image)
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]


    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
   
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def count(thresholded, segmented):
    convex_hull = cv2.convexHull(segmented)

    extreme_top = tuple(convex_hull[convex_hull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(convex_hull[convex_hull[:, :, 1].argmax()][0])
    extreme_left = tuple(convex_hull[convex_hull[:, :, 0].argmin()][0])
    extreme_right = tuple(convex_hull[convex_hull[:, :, 0].argmax()][0])

    cX = (extreme_left[0] + extreme_right[0]) / 2;
    cY = (extreme_top[1] + extreme_bottom[1]) /2;

    dist = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_dist = dist[dist.argmax()]

    radius = int(0.8 * maximum_dist)

    circumference = (2 * np.pi * radius)

    cir_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(cir_roi, (int(cX), int(cY)), radius, 255, 1)
    cir_roi = cv2.bitwise_and(thresholded, thresholded, mask=cir_roi)

    (_, cnts, _) = cv2.findContours(cir_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# initalize the finger count
    count = 0

	# loop through the contours found
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

	# increment the count of fingers only if -
	# 1. The contour region is not the wrist (bottom area)
	# 2. The number of points along the contour does not exceed
	#     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count




# main function

if __name__ == '__main__':
    aWeight = 0.5
    cap = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 255, 590

    num_frams = 0;
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=700)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if num_frams < 30:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray, 25)
            if hand is not None:
                (thresholded, segmented) = hand
                fing_count = count(thresholded, segmented)

                print(fing_count)
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Threholded", thresholded)
        
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frams += 1
        cv2.imshow("Video Feed", clone)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
