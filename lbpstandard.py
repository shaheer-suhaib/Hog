from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time

def padding(img, pad):
    img[:pad, :] = 0
    img[-pad:, :] = 0
    img[:, :pad] = 0
    img[:, -pad:] = 0
    return img

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

def lbp(img, filter):
    img2 = img.copy()
    rows, cols = img.shape
    frows, fcols = filter.shape
    for i in range(rows - frows + 1):
        for j in range(cols - fcols + 1):
            window = img[i:i + frows, j:j + fcols]
            img2[i, j] = lbpAlgo(window)
    return img2.astype(np.uint8)

def main():
    # Initialize Pi Camera
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (320, 240)  # Lower resolution = faster processing
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    time.sleep(1)  # Warm-up time

    while True:
        frame = picam2.capture_array()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        padded = padding(gray.copy(), 1)
        filter = np.zeros((3, 3), dtype=np.uint8)
        lbp_frame = lbp(padded, filter)

        cv.imshow("Original", gray)
        cv.imshow("LBP Output", lbp_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
