from tkinter.messagebox import showinfo
import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt
import sys
from Solver import *
import timeit

def load_picture(pic1, pic2, downscale=False, show_image=False):
    imgL = cv2.imread(pic1, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(pic2, cv2.IMREAD_GRAYSCALE)
    imgL_small = cv2.resize(imgL, (0,0), fx=1/7, fy=1/7) 
    imgR_small = cv2.resize(imgR, (0,0), fx=1/7, fy=1/7) 
    if(show_image):
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1, 2, 1)
        plt.imshow(imgL_small, cmap='gray')
        plt.axis('off')
        fig.add_subplot(1, 2, 2)
        plt.imshow(imgR_small, cmap='gray')
        plt.axis('off')
        plt.show()
    if(downscale):
        return imgL_small, imgR_small
    else:
        return imgL, imgR

if __name__=="__main__":
    folder = sys.argv[1]
    limg, rimg = load_picture(folder+'/im0.png', folder+'/im1.png', downscale=False)
    row0 = int(sys.argv[2])
    row1 = int(sys.argv[3])
    print("running with ", folder, " for row", row0, " to ", row1)
    numdis = int(sys.argv[5])
    testSolver = Solver(rimg[row0:row1], limg[row0:row1], 4, numdis,None ,d=2, cutoff=30)
    start = timeit.default_timer()
    testSolver.run()
    stop = timeit.default_timer()
    outfolder = sys.argv[4]
    np.save(os.path.join(outfolder,str(row0)+"to"+str(row1)+".npy"), testSolver.d_left)
    print("runtime", stop-start)
    plt.imshow(testSolver.d_left)
    plt.colorbar()
    plt.show()
    
