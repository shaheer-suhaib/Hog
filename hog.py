import cv2
import numpy as np


sobelx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

sobely = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])


img = cv2.imread('Lab 10/image4.png', cv2.IMREAD_GRAYSCALE)

winsize = (img.shape[0]//2, img.shape[1]//2)
blocksize = (winsize[0]//8, winsize[1]//8)
cellsize = (blocksize[0]//2, blocksize[1]//2)

imagehist = []

for i in range(0, img.shape[0], winsize[0]):
    for j in range(0, img.shape[1], winsize[1]):
        win = img[i:i+winsize[0], j:j+winsize[1]]
        winhist = []
        for k in range(0, winsize[0], blocksize[0]):
            for l in range(0, winsize[1], blocksize[1]): 
                block = win[k:k+blocksize[0], l:l+blocksize[1]]
                blockhist = []         
                for m in range(0, blocksize[0], cellsize[0]):
                    for n in range(0, blocksize[1], cellsize[1]):
                        hist = np.zeros(6)
                        cell = block[m:m+cellsize[0], n:n+cellsize[1]]
                        gx = cv2.filter2D(cell, -1, sobelx)
                        gy = cv2.filter2D(cell, -1, sobely)
                        mag = np.sqrt(gx**2 + gy**2)
                        mag = np.uint8(mag)
                        phases = np.arctan2(gy, gx) * 180 / np.pi
                        phases[phases < 0] += 180
                        for p in range(0, 180, 30):
                            hist[p//30] = np.sum(mag[(phases >= p) & (phases < p+30)])                          
                        blockhist.extend(hist)
                blockhist = np.array(blockhist)
                blockhist = blockhist.flatten()
                winhist.extend(blockhist)
            
        winhist = np.array(winhist)        
        winhist = winhist.flatten()
        imagehist.extend(winhist)
    

print(imagehist[:100])
