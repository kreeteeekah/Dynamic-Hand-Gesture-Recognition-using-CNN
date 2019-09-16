import cv2

def main():
    
    for i in range(1,6):
        #Ismai images ka naming numerically linear sa hona chahiye
        imgpath = "C:\\Users\\MUJ\\Desktop\\Minor Project\\test1\\"+"palm"+str(i)+".png"
    
        outpath = "C:\\Users\\MUJ\\Desktop\\Minor Project\\test2\\"+"palm"+str(i)+".png"
        
        #While reading the image it is read as gray scaled
        #Change below line to do any operation on the image
        img = cv2.imread(imgpath, 0)

        #The parameter 'op' is the name of the window in which the image is opened
        #cv2.imshow('op',img)
        cv2.imwrite(outpath, img)
    
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()