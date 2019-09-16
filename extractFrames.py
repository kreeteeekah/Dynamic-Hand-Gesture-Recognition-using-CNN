# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 18:45:03 2019

@author: MUJ
"""

# Program To Read video 
# and Extract Frames 
import cv2 

# Function to extract frames 
def FrameCapture(path): 
	
	# Path to video file 
	vidObj = cv2.VideoCapture(path) 

	# Used as counter variable 
	count = 0

	# checks whether frames were extracted 
	success = 1

	while success: 

		# vidObj object calls read 
		# function extract frames 
		success, image = vidObj.read() 

		# Saves the frames with frame-count 
		cv2.imwrite("frame%d.png" % count, image) 

		count += 1

# Driver Code 
if __name__ == '__main__': 

	# Calling the function 
	FrameCapture("F:\leap dataset\leapGestRecog\IMG-4717.mp4") 
