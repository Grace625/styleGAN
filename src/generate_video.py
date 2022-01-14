import cv2
import numpy as np

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('video_ffhq.avi', fourcc, 200, (1302, 652))

for j in range(0, 2001):
    img = cv2.imread('output/sample_' + str(j) + '.png')
    video.write(img)

cv2.destroyAllWindows()
video.release()