from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time

def padding(img, pad):
    img_padded = img.copy()
    img_padded[:pad, :] = 0
    img_padded[-pad:, :] = 0
    img_padded[:, :pad] = 0
    img_padded[:, -pad:] = 0
    return img_padded

def lbpAlgo(window):
    gc = window[1, 1]
    neighbors = [
        window[0, 0], window[0, 1], window[0, 2],
        window[1, 2], window[2, 2], window[2, 1],
        window[2, 0], window[1, 0]
    ]
    binary = [1 if p > gc else 0 for p in neighbors]
    weights = [1 << i for i in range(8)]
    return sum(binary[i] * weights[i] for i in range(8))

def lbp(img):
    img2 = img.copy()
    rows, cols = img.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = img[i-1:i+2, j-1:j+2]
            img2[i, j] = lbpAlgo(window)
    return img2.astype(np.uint8)

def main():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (320, 240)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    time.sleep(1)

    last_capture_time = time.time()  # Initialize time tracking
    capture_interval = 3  # Capture every 3 seconds

    while True:
        frame = picam2.capture_array()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        # Capture frame and compute LBP every few seconds
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            padded = padding(gray, 1)
            lbp_frame = lbp(padded)

            # Display both original and LBP frames
            cv.imshow("Original", gray)
            cv.imshow("LBP Output", lbp_frame)

            # Update last capture time
            last_capture_time = current_time

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
