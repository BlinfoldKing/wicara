import numpy as np
import cv2

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
        return (cnts, segmented)


# main function

if __name__ == '__main__':
    aWeight = 0,5
    cap = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 255, 590

    num_frams = 0;

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) 
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        if num_frams < 30:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                print(thresholded)
                cv2.imshow("Threholded", thresholded)
        
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frams += 1
        cv2.imshow("Video Feed", clone)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
