import cv2

def main():
    
    for i in range(1,3):
        #for reading the images from dataset
        imgpath = "F:\\leap dataset\\leapGestRecog\\binarized dataset\\single prediction\\"+"c_or_palm"+str(i)+".png"
    
        outpath = "F:\\leap dataset\\leapGestRecog\\binarized dataset\\binarized_prediction\\"+"c_or_palm"+str(i)+".png"
        
        #While reading the image it is read as gray scaled
        #Change below line to do any operation on the image
        img = cv2.imread(imgpath, 0)
        
        median = cv2.medianBlur(img,5)
        threshold = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        #The parameter 'op' is the name of the window in which the image is opened
        #cv2.imshow('op',img)
        #to resize the image to 28x28 pixel
        write_img = cv2.resize(threshold, (128,128))

        cv2.imwrite(outpath, write_img)
    
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
