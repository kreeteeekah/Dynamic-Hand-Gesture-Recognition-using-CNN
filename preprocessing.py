import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    
    for i in range(1,5):
        
        #for reading the images in the dataset
        imgpath = "C:\\Users\\MUJ\\Desktop\\Minor Project\\test1\\"+"palm"+str(i)+".png"
    
        outpath = "C:\\Users\\MUJ\\Desktop\\Minor Project\\test2\\"+"palm"+str(i)+".png"
        
        #While reading the image it is read as gray scaled
        #Change below line to do any operation on the image
        img = cv2.imread(imgpath, 0)
        median = cv2.medianBlur(img,5)
        threshold = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        
        #cv2.imshow('op',img)
        plt.show(threshold)
        cv2.imwrite(outpath, threshold)
    
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
