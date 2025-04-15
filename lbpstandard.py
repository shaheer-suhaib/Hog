import cv2 as cv
import numpy as np

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
    return img2

def main():
    cap = cv.VideoCapture(0)

    # Reduce resolution for performance
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not captured.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        padded = padding(gray, 1)
        filter = np.zeros((3, 3), dtype=np.uint8)
        lbp_frame = lbp(padded, filter)

        cv.imshow('Original', gray)
        cv.imshow('LBP Frame', lbp_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
