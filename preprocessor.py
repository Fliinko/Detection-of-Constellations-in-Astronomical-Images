#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
#%%
#PREPROCESSING THE IMAGE
print("Preprocessing Images...")
startpre = datetime.now()
for x in os.listdir("Meddled"):
	print("Preprocessing: ", x)
	image = cv2.imread("Meddled/" + x)

	#  constants
	BINARY_THRESHOLD = 20
	CONNECTIVITY = 4
	DRAW_CIRCLE_RADIUS = 4

	#  convert to gray
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#  extract edges
	binary_image = cv2.Laplacian(gray_image, cv2.CV_8UC1)

	#  fill in the holes between edges with dilation
	dilated_image = cv2.dilate(binary_image, np.ones((5, 5)))

	#  threshold the black/ non-black areas
	_, thresh = cv2.threshold(dilated_image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

	#  find connected components
	components = cv2.connectedComponentsWithStats(thresh, CONNECTIVITY, cv2.CV_32S)

	#  draw circles around center of components
	#see connectedComponentsWithStats function for attributes of components variable
	centers = components[3]
	for center in centers:
		cv2.circle(thresh, (int(center[0]), int(center[1])), DRAW_CIRCLE_RADIUS, (255), thickness=-1)
		
	
	cv2.imwrite("Preprocessed/res" + x, thresh)
	#cv2.imshow("result", thresh)
	cv2.waitKey(0)

	print("Finished Preprocessing: ", x)

endpre = datetime.now()
print("[MSG] Finished Preprocessing All Images")
print(endpre-startpre)
#%%
def Lines(image, contours, x, y):
    
	canvas = image
	for i in range(0, len(contours)):
				
		c = contours[i]
		size = cv2.contourArea(c)
		if (size < 1000):
				M = cv2.moments(c)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])

				x.append(cX)
				y.append(cY)

	xy = list(zip(x,y))
	xy = np.array(xy)

	for i in range(0, len(xy)):
    		
		x1 = xy[i,0]
		y1 = xy[i,1]
		distance = 0
		secondx = []
		secondy = []
		dist_listappend = []
		sort = []   
		for j in range(0, len(xy)):      
			if i == j:
				pass     
			else:
				x2 = xy[j,0]
				y2 = xy[j,1]
				distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
				secondx.append(x2)
				secondy.append(y2)
				dist_listappend.append(distance)      

		secondxy = list(zip(dist_listappend,secondx,secondy))
		sort = sorted(secondxy, key=lambda second: second[0])
		sort = np.array(sort)
		cv2.line(canvas, (x1,y1), (int(sort[0,1]), int(sort[0,2])), (0,0,255), 2)
				
#%%
#DETECTING BRIGHT PIXELS
from imutils import contours
from skimage import measure
import imutils

print("[MSG] Starting Brightest Star Detection...\n")
start = datetime.now()
for img in os.listdir("Preprocessed"):
    
	print("Preprocessing: ", img)
	counter = 1
	imge = cv2.imread("Preprocessed/" + img)
	gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)

	thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=4)

	# perform a connected component analysis on the thresholded
	# image, then initialize a mask to store only the "large" components
	labels = measure.label(thresh, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")
	# loop over the unique components
	for label in np.unique(labels):
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels > 255:
			mask = cv2.add(mask, labelMask)

	# find the contours in the mask, then sort them from left to right
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	listx = []
	listy = []
	for method in ("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
		try:		
			(cnts, boundingBoxes) = contours.sort_contours(cnts, method = method)
		except Exception as e:
			print(str(e))
	# loop over the contours
	for (i, c) in enumerate(cnts):
		# draw the bright spot on the image
		start_point = (i, c)
		end_point = (i+1, c+1)
		(x, y, w, h) = cv2.boundingRect(c)
		((cX, cY), radius) = cv2.minEnclosingCircle(c)
		cv2.circle(imge, (int(cX), int(cY)), int(radius),
			(0, 0, 255), 7)
		Lines(imge, cnts, listx, listy)
		cv2.putText(imge, "#{}".format(i + 1), (x, y - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	# show the output image
	cv2.imwrite("Matchable/" + img + ".png", imge, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	#cv2.imshow("Image", image)
	cv2.waitKey(0)
	counter += 1 
	print("Finished Detection of: ", img)

end = datetime.now()
print("\n[MSG] Finished Detection of All Images in: \n")
print(end-start)

# %%
