def process_image():
	# Block 1

	import cv2
	import numpy as np
	import matplotlib.pyplot as plt
	#matplotlib inline
	import pandas as pd
	import os
	from scipy import ndimage
	import math
	import keras
	import ast
	import operator as op
	import re
	from tensorflow.keras.preprocessing.image import ImageDataGenerator
	#Suppressing warning
	def warn(*args, **kwargs):
			pass
	import warnings
	warnings.warn = warn

	import imutils

	# Block 2

	#Global Variable
	dict_clean_img = {} #BINARY IMAGE DICTIONAY
	dict_img = {} #ORIGINAL IMAGE DICTIONARY

	#Keras support channel first (1,28,28) only
	keras.backend.set_image_data_format("channels_first")

	# Block 3

	#declaring global variables for evaluation
	global cleaned_orig
	global num1
	global num2
	global operator
	global is_valid
	is_valid=True
	num1=num2=operator=""
	wrong_lines=[]

	# Block 4

	try:
			model = keras.models.load_model('DCNN_10AD_sy.h5', compile=False)

	except Exception as e:
			print('Model could not be loaded',e)

	# Block 5

	#output image manipulation functions
	#1 def invalidImage()
	#2 def correctImage(output_img)
	#3 def wrongImage(output_img,wrong_lines)

	#1
	def invalidImage():
		output_img = cv2.imread("invalid_input.png")
		font = cv2.FONT_HERSHEY_SIMPLEX
		bottom_left = (120, 220)
		fontScale = 1
		color = (230,100,255)
		thickness = 3
		output_img = cv2.putText(output_img, '( ' + num1 + ' )' + "  " + operator + "  " +  '( ' +  num2 + ' )' , bottom_left, font,fontScale, color, thickness, cv2.LINE_AA)
		return output_img

	#2
	def correctImage(output_img):
		#write correctly solved
		font = cv2.FONT_HERSHEY_SIMPLEX
		bottom_left_x = w-320
		bottom_left_y = 60
		bottom_left = ( bottom_left_x , bottom_left_y ) 

		fontScale = 1.2
		color = (230,100,255)
		thickness = 3
		output_img = cv2.putText(output_img, "Correctly Solved" , bottom_left, font,fontScale, color, thickness, cv2.LINE_AA)
		return output_img

	#3
	def wrongImage(output_img,wrong_lines):
		col1=(0,0,255)
		col2=(255,0,0)
		for i in range(len(wrong_lines)):
			Y1=wrong_lines[i][0]
			Y2=wrong_lines[i][1]
			correct=wrong_lines[i][2]
			X1=0+15
			X2=w-15
			if i%2!=0:
				col=col1
			else:
				col=col2
			cv2.rectangle(output_img ,(X1,Y1),(X2,Y2),col,4)
			#write correct value in rectangular box
			font = cv2.FONT_HERSHEY_SIMPLEX
			bottom_left_x = int( X2 - (X2-X1)/4 )
			bottom_left_y = int(Y1+35)
			bottom_left = ( bottom_left_x , bottom_left_y ) 

			fontScale = 1.2
			color = (230,100,255)
			thickness = 3
			output_img = cv2.putText(output_img, correct , bottom_left, font,fontScale, color, thickness, cv2.LINE_AA)
		return output_img

	# Block 6

	#functions
	#1 def sort_contours(cnts, method="left-to-right"):
	#2 def getBestShift(img):
	#3 def shift(img,sx,sy):
	#4 def predict(img,x1,y1,x2,y2, proba = False, acc_thresh = 0.60):
	#5 def process_img (gray, resize_flag = 1, preproc = 0):
	#6 def find_good_contours_thres(conts, alpha = 0.002):
	#7 def extract_line(image, beta=0.4, alpha=0.002, show = True):
	#8 def draw_contour(image, c, i):
	#9 def text_segment(Y1,Y2,X1,X2,box_num,line_name, dict_clean = dict_clean_img,acc_thresh = 0.60, show = True):
	#10 def shadowRemoval(img):
	#11 def cleanImage(img):
	#12 def rotate(filtered_img):
	#13 def bbox(img):
	#14 def isLineExist(img):
	#15 def preprocessImage(test_img):
	#16 def lineExtraction1(test_img):
	#17 def removeDashLines(df_lines):
	#18 def lineExtraction2(list_chars):

	#1
	def sort_contours(cnts, method="left-to-right"):
		'''
		sort_contours : Function to sort contours
		argument:
				cnts (array): image contours
				method(string) : sorting direction
		output:
				cnts(list): sorted contours
				boundingBoxes(list): bounding boxes
		'''
		# initialize the reverse flag and sort index
		reverse = False
		i = 0

		# handle if we need to sort in reverse
		if method == "right-to-left" or method == "bottom-to-top":
				reverse = True

		# handle if we are sorting against the y-coordinate rather than
		# the x-coordinate of the bounding box
		if method == "top-to-bottom" or method == "bottom-to-top":
				i = 1

		# construct the list of bounding boxes and sort them from top to
		# bottom
		boundingBoxes = [cv2.boundingRect(c) for c in cnts]
		(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
				key=lambda b:b[1][i], reverse=reverse))

		# return the list of sorted contours and bounding boxes
		return (cnts, boundingBoxes)

	#2
	def getBestShift(img):
		'''
		getBestShift : Function to calculate centre of mass and get the best shifts
		argument:
				img (array) : gray scale image
		output:
				shiftx, shifty: x,y shift direction
		'''
		cy,cx = ndimage.measurements.center_of_mass(img)
		rows,cols = img.shape
		shiftx = np.round(cols/2.0-cx).astype(int)
		shifty = np.round(rows/2.0-cy).astype(int)

		return shiftx,shifty

	#3
	def shift(img,sx,sy):
		'''
		Shift : Function to shift the image in given direction 
		argument:
				img (array) : gray scale image
				sx, sy      : x, y direction
		output:
				shifted : shifted image
		'''
		rows,cols = img.shape
		M = np.float32([[1,0,sx],[0,1,sy]])
		shifted = cv2.warpAffine(img,M,(cols,rows))
		return shifted

	# #%%
	# #Data Generator using tensorflow method
	# train_datagen = ImageDataGenerator(   
	#     data_format='channels_first',
	#     zca_whitening = True,
	#     rotation_range = 0.2)

	#4
	#%%
	def predict(img,x1,y1,x2,y2, proba = False, acc_thresh = 0.60):

		'''
		predict  : Function to predict the character
		argument:
				x1,y1(int,int)    : Top left corner point
				x2,y2(int,int)    : Bottom right corner point
				acc_thresh(0-1)   : Probability threshold for calling model_robusta
				proba(bool)       : If probability values is wanted in return
		output:
				c[index](int) : predicted character 
			
			'''
		gray = img[y1:y2, x1:x2]

		# Steps to remove noises in image due to cropping
		temp = gray.copy()
		
		kernel_temp = np.ones((3,3), np.uint8) 
		temp_tmp = cv2.dilate(temp, kernel_temp, iterations=3)
		
		# Find the contours -  To check whether its disjoint character or noise

		contours_tmp,hierarchy = cv2.findContours(temp_tmp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
				
		if(len(contours_tmp) > 1):
				# Find the contours
				contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
				#Creating a mask of only zeros  
				mask = np.ones(gray.shape[:2], dtype="uint8") * 0
				# Find the index of the largest contour
				areas = [cv2.contourArea(c) for c in contours]
				max_index = np.argmax(areas)
				cnt=contours[max_index]
				
				cv2.drawContours(mask, [cnt], -1, 255, -1)
				#Drawing those contours which are noises and then taking bitwise and
				gray = cv2.bitwise_and(temp, temp, mask=mask)
				
		grayN = process_img (gray, resize_flag = 0)
		
		classes = model.predict(grayN, batch_size=2)
		ind = np.argmax(classes)
		c = ['0','1','2','3','4','5','6','7','8','9','+','-','*','(',')']

		
		if (proba == True):
				return classes[0][ind]
		
		return c[ind]

	#5
	#%%
	def process_img (gray, resize_flag = 1, preproc = 0):
			'''
			process_img  : Function to pre process image for prediction
			argument:
					gray (Matrix (np.uint8))  : image of character to be resized and processed
					resize_flag               : method used for resizing image
					preproc (method [bool])   : 0 : No erosion DIlation, 1 : Erosion, Dilation
			output:
					grayS (Matrix (0-1))      : Normalised image of character resized and processed
			
			'''    
			gray = gray.copy()
			
			#Image Pre Processing
			if (preproc == 0):
					gray = cv2.GaussianBlur(gray,(7,7),0)
			else :
					kernel = np.ones((3,3), np.uint8)
					gray = cv2.dilate(gray, kernel, iterations=1)    
					gray = cv2.GaussianBlur(gray,(5,5),1)
					gray = cv2.dilate(gray, kernel, iterations=2)
					gray = cv2.erode(gray, kernel,iterations=2)    
			
			#Removing rows and columns where all the pixels are black
			while np.sum(gray[0]) == 0:
					gray = gray[1:]

			while np.sum(gray[:,0]) == 0:
					gray = np.delete(gray,0,1)

			while np.sum(gray[-1]) == 0:
					gray = gray[:-1]

			while np.sum(gray[:,-1]) == 0:
					gray = np.delete(gray,-1,1)

			rows,cols = gray.shape
			
			if(resize_flag) == 1:
					interpolation=cv2.INTER_AREA
			else:
					interpolation=cv2.INTER_CUBIC
			# Making the aspect ratio same before re-sizing
			if rows > cols:
					factor = 20.0/rows
					rows = 20
					cols = int(round(cols*factor))
					# first cols than rows
					gray = cv2.resize(gray, (cols,rows),interpolation=interpolation)
			else:
					factor = 20.0/cols
					cols = 20
					rows = int(round(rows*factor))
					# first cols than rows
					gray = cv2.resize(gray, (cols, rows),interpolation=interpolation)
		
			# Padding to a 28 * 28 image
			colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
			rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
			gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
			
			# Get the best shifts
			shiftx,shifty = getBestShift(gray)
			shifted = shift(gray,shiftx,shifty)
			grayS = shifted
			grayS = grayS.reshape(1,1,28,28)
			
			#Normalize the image
			grayS = grayS/255
			
			return grayS


	'''
	Line Detection
	'''
	#6
	def find_good_contours_thres(conts, alpha = 0.002):
			'''
			Function to find threshold of good contours on basis of 10% of maximum area
			Input: Contours, threshold for removing noises
			Output: Contour area threshold
			
			For image dim 3307*4676
			alpha(text_segment) = 0.01
			alpha(extract_line) = 0.002
			'''
			#Calculating areas of contours and appending them to a list
			areas = []
			
			for c in conts:
					areas.append([cv2.contourArea(c)**2])
			#alpha is controlling paramter    
			thres = alpha * max(areas)[0]
			
			return thres
		

	#7
	def extract_line(image, beta=0.4, alpha=0.002, show = True):
			'''
			Function to extracts the line from the image   
			Assumption : Sufficient gap b/w lines
			
			argument:
					img (array): image array
					beta (0-1) : Parameter to differentiate line
					alpha (0-1) : Parameter to select good contours
					show(bool) : to show figures or not
			output:
					uppers[diff_index]  : Upper points (x,y)
					lowers[diff_index]  : lower points(x,y)
			'''
			global is_valid
			img = image.copy()
			H,W = img.shape[:2]
			h5 = int(.02 * H)
			w5 = int(.02 * W)
			img[:h5,:] = [255,255,255]
			img[-h5:,:] = [255,255,255]
			img[:,:w5] = [255,255,255]
			img[:,-w5:] = [255,255,255]
			
			#Converting image to gray
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			#Binary thresholding and inverting at 127
			th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
			#Selecting elliptical element for dilation    
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
			dilation = cv2.dilate(threshed,kernel,iterations = 1)
			#Saving a copy of dilated image for taking bitwise_and operation
			temp = dilation.copy()
			
			#////////////////////////////////////////////////////////////////////////////////
			# Find the contours
			if(cv2.__version__ == '3.3.1'): 
					xyz,contours,hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			else:
					contours,hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
					
			cont_thresh = find_good_contours_thres(contours, alpha=alpha)


			#Creating a mask of only ones    
			mask = np.ones(dilation.shape[:2], dtype="uint8") * 255

			#Drawing those contours which are noises and then taking bitwise and
			for c in contours:
					if( cv2.contourArea(c)**2 < cont_thresh):
							cv2.drawContours(mask, [c], -1, 0, -1)
			
			cleaned_img = cv2.bitwise_and(temp, temp, mask=mask)
			
			#Dilating the cleaned image for better detection of line in cases where
			#exponents are little up the line
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
			dil_cleaned_img = cv2.dilate(cleaned_img,kernel,iterations = 10)
			
			
			#Getting back the cleaned original image without noise
			cleaned_orig = cv2.erode(cleaned_img, kernel, iterations=1) 
			
			# /////////////////////////////////////////////////////////////////////////////////////////
			##find and draw the upper and lower boundary of each lines
			hist = cv2.reduce(dil_cleaned_img,1, cv2.REDUCE_AVG).reshape(-1)

			th = 1
			H,W = img.shape[:2]

			uppers = np.array([y for y in range(H-1) if hist[y]<=th and hist[y+1]>th])
			lowers = np.array([y for y in range(H-1) if hist[y]>th and hist[y+1]<=th])
			
			if( (len(uppers)!=len(lowers)) or (len(uppers)==0) or (len(lowers)==0) ):
				is_valid=False
				return cleaned_orig,uppers,lowers
			


			#print("uppers  ")
			print(uppers ,"uppers")
			#print("lowers  ")
			print(lowers,"lowers")



			diff_1 = np.array([j-i for i,j in zip(uppers,lowers)])
			diff_index_1 = np.array([True if j > beta*(np.mean(diff_1)-np.std(diff_1)) else False for j in diff_1 ])

			
			print(diff_1 ,"diff_1 " )
			print(diff_index_1, "diff_index_1  " )

			
			uppers = uppers[diff_index_1]
			lowers = lowers[diff_index_1]
			

			
			#Extending uppers and lowers indexes to avoid cutting of chars of lines
			#Extended more uppers by 33% as exponential might lie above 
			# uppers[1:] = [i-int(j)/10 for i,j in zip(uppers[1:], diff_1[1:])]
			# lowers[:-1] = [i+int(j)/10 for i,j in zip(lowers[:-1], diff_1[:-1])]
			
			
			diff_2 = np.array([j-i for i,j in zip(uppers,lowers)])
			diff_index_2 = np.array([True]*len(uppers))
			
			
	

			print(diff_2,"diff_2")
			print(diff_index_2,"diff_index_2")

			diff_index = diff_index_2

			
			cleaned_orig_rec = cv2.cvtColor(cleaned_orig, cv2.COLOR_GRAY2BGR)
			

			#For changing color of intermediate lines, keeping count
			col_ct = 0
			
			for left,right in zip(uppers[diff_index], lowers[diff_index]):
					#print(left,right)
					col1 = (255,150,80)
					col2 = (0,255,0)
					if(col_ct % 2 == 0):
							col= col1
					else: 
							col=col2
					cv2.rectangle(cleaned_orig_rec ,(0+10,left),(W-15,right),col,4)
					col_ct += 1
					
			# if(show == True):
			#     fig0 = plt.figure(figsize=(15,12))
			#     ax1 = fig0.add_subplot(1,3,1)
			#     ax1.set_title('Original Image')
			#     ax1.imshow(img)
			#     ax1.axis('off')
					
			#     ax2 = fig0.add_subplot(1,3,2)
			#     ax2.set_title('Cleaned Image')
			#     ax2.imshow(cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2RGB))
			#     ax2.axis('off')
					
			#     ax3 = fig0.add_subplot(1,3,3)
			#     ax3.set_title('Noises')
			#     ax3.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
			#     ax3.axis('off')
					
			#     fig0.suptitle('Denoising')
			#     plt.show()
			
			#     fig1 = plt.figure(figsize=(15,10))
			#     fig1.suptitle('Line Detection')
			#     ax1 = fig1.add_subplot(1,2,1)
			#     ax1.axis("off")
			#     ax1.imshow(cv2.cvtColor(cleaned_orig,cv2.COLOR_BGR2RGB))
					
			#     ax2 = fig1.add_subplot(1,2,2)    
			#     ax2.axis("off")
			#     ax2.imshow(cv2.cvtColor(cleaned_orig_rec, cv2.COLOR_BGR2RGB))
					
			#     plt.show()
			
			return cleaned_orig, uppers[diff_index], lowers[diff_index]


	#8
	def draw_contour(image, c, i):
		# compute the center of the contour area and draw a circle
		# representing the center
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	
		# draw the countour number on the image
		cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
	
		# return the image with the contour number drawn on it
		return image

	#9
	def text_segment(Y1,Y2,X1,X2,box_num,line_name, dict_clean = dict_clean_img,acc_thresh = 0.60, show = True):
			'''
			text_segment : Function to segment the characters
			Input:
					Box coordinates -X1,Y1,X2,Y2
					box_num - name of box
					line_name - name of line
					model - Deep Learning model to be used for predictionl̥
					dict_clean - dictionary of clean box images
			Output :
					box_num - name of box
					line_name -name of line
					df_char - Dataframe of characters of that particular line
			'''
			img = dict_clean[box_num][Y1:Y2,X1:X2].copy()
			L_H = Y2-Y1
			## apply some dilation and erosion to join the gaps
			#Selecting elliptical element for dilation    
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
			dilation = cv2.dilate(img,kernel,iterations = 2)
			erosion = cv2.erode(dilation,kernel,iterations = 1)
			
			# Find the contours
			if(cv2.__version__ == '3.3.1'):
					xyz,contours,hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			else:
					contours,hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
					
			print("contour len " ,len(contours))

			ct_th = find_good_contours_thres(contours, alpha=0.002)
			cnts = []
			for c in contours:     
					if( cv2.contourArea(c)**2 > ct_th ):
							cnts.append(c)

			print("good contour len " ,len(cnts))

			contours_sorted,bounding_boxes = sort_contours(cnts,method="left-to-right")
			
			
			char_locs = []
			
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
			
			i = 0
			#char_type =[]
			while i in range(0, len(contours_sorted)):
							x,y,w,h = bounding_boxes[i]
							#print("printing bounding boxes")
							#print(x," ",y," ",w," ",h)
							if w>9*h :
								print("w>8*h ", i," " ,w, " -- ",h)
								i=i+1
								continue
							char_locs.append([x-2,y+Y1-2,x+w+1,y+h+Y1+1,w*h]) #Normalised location of char w.r.t box image
							
							cv2.rectangle(img,(x,y),(x+w,y+h),(255,150,80),2)
							i=i+1

			df_char = pd.DataFrame(char_locs)
			df_char.columns=['X1','Y1','X2','Y2','area']
			df_char['pred'] = df_char.apply(lambda c: predict(dict_clean[box_num],c['X1'],c['Y1'], c['X2'], c['Y2'], acc_thresh=acc_thresh), axis=1 )
			df_char['pred_proba'] = df_char.apply(lambda c: predict(dict_clean[box_num],c['X1'], c['Y1'],c['X2'], c['Y2'], proba=True, acc_thresh=acc_thresh), axis=1 )
			df_char.apply(lambda c: cv2.putText(img, c['pred'], (c['X1']-10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0), 6, cv2.LINE_AA), axis=1) 
			df_char['line_name'] = line_name
			df_char['box_num'] = box_num

			# if(show == True): 
			#   plt.figure(figsize=(15,8))   

			#   plt.axis("on")
			#   plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			#   plt.show()
					
			return [box_num,line_name,df_char]

	#10
	def shadowRemoval(img):
		img = org_img.copy()
		print(img.shape)
		rgb_planes = cv2.split(img)

		result_planes = []
		result_norm_planes = []
		for plane in rgb_planes:
				#n cv2.dilate(plane, np.ones((21,21), np.uint8)) if we increase (21,21) value then image written element become dark  and shadows also started coming 
				#low value of cv2.medianBlur(dilated_img, 19) increase some edge fines gaps ,,but if we increase then after certain value no change in image 
				#initial cv2.dilate(plane, np.ones((7,7), np.uint8)) , cv2.medianBlur(dilated_img, 21)

				dilated_img = cv2.dilate(plane, np.ones((17,17), np.uint8))
				bg_img = cv2.medianBlur(dilated_img, 19)
				diff_img = 255 - cv2.absdiff(plane, bg_img)
				norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
				result_planes.append(diff_img)
				result_norm_planes.append(norm_img)

		result = cv2.merge(result_planes)
		result_norm = cv2.merge(result_norm_planes)

		cv2.imwrite('shadowless1.png', result)
		cv2.imwrite('shadowless2.png', result_norm)
		shadowless1=cv2.imread('shadowless1.png')
		shadowless2=cv2.imread('shadowless2.png')


		image_gray1 = cv2.cvtColor(shadowless1, cv2.COLOR_BGR2GRAY)
		image_gray2 = cv2.cvtColor(shadowless2, cv2.COLOR_BGR2GRAY)
		thresh, binary_image1 = cv2.threshold(image_gray1, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		thresh, binary_image2 = cv2.threshold(image_gray2, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


		kernel = np.ones((1,1), np.uint8)
		dilated_img2 = cv2.dilate(binary_image2, kernel, iterations=1)

		# plt.figure(figsize=(35,35))
		# plt.subplot(1,5,1)
		# plt.imshow( cv2.cvtColor(shadowless1, cv2.COLOR_BGR2RGB))
		# plt.subplot(1,5,2)
		# plt.imshow( cv2.cvtColor(shadowless2, cv2.COLOR_BGR2RGB))
		# plt.subplot(1,5,3)
		# plt.imshow( cv2.cvtColor(binary_image1, cv2.COLOR_BGR2RGB))
		# plt.subplot(1,5,4)
		# plt.imshow( cv2.cvtColor(binary_image2, cv2.COLOR_BGR2RGB))

		dilated_img2 = cv2.copyMakeBorder(dilated_img2, 5, 5, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
		# plt.subplot(1,5,5)
		# plt.imshow( cv2.cvtColor(dilated_img2, cv2.COLOR_BGR2RGB))


		test_img=cv2.cvtColor( cv2.bitwise_not(dilated_img2), cv2.COLOR_GRAY2RGB)  
		return test_img

	#11
	def cleanImage(img):
		alpha=0.005
		beta=0.4
		#Converting image to gray
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#Binary thresholding and inverting at 127
		th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
		#Selecting elliptical element for dilation    
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		dilation = cv2.dilate(threshed,kernel,iterations = 1)
		#Saving a copy of dilated image for taking bitwise_and operation
		temp = dilation.copy()
		
		# Find the contours
		if(cv2.__version__ == '3.3.1'): 
				xyz,contours,hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		else:
				contours,hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
				
		cont_thresh = find_good_contours_thres(contours, alpha=alpha)


		#Creating a mask of only ones    
		mask = np.ones(dilation.shape[:2], dtype="uint8") * 255

		#Drawing those contours which are noises and then taking bitwise and
		for c in contours:
				if( cv2.contourArea(c)**2 < cont_thresh):
						cv2.drawContours(mask, [c], -1, 0, -1)
		
		cleaned_img = cv2.bitwise_and(temp, temp, mask=mask)
		
		#Dilating the cleaned image for better detection of line in cases where
		#exponents are little up the line
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		dil_cleaned_img = cv2.dilate(cleaned_img,kernel,iterations = 10)
		
		
		#Getting back the cleaned original image without noise
		cleaned_orig = cv2.erode(cleaned_img, kernel, iterations=1) 
		#plt.figure(figsize=(8,8))
		# plt.imshow(cleaned_orig)
		return cleaned_orig

	#12
	def rotate(filtered_img):
			img=filtered_img.copy()
			if(cv2.__version__ == '3.3.1'):
							xyz,contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			else:
					contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			boundingBoxes = [cv2.boundingRect(c) for c in contours]
			diag=([math.sqrt(l**2+b**2) for x,y,l,b in boundingBoxes])
			mx=max(diag)
			mx_ind=diag.index(mx)
			print(mx,mx_ind)
			#plt.plot(diag)
			#plt.show()
			x,y,l,b=boundingBoxes[mx_ind]
			ang=math.atan(-b/l)
			h_line=contours[mx_ind]
			sign=1
			left_point_index=list(h_line[:,0,0]).index(min(h_line[:,0,0]))
			if(h_line[left_point_index][0][1]-y>b//3):sign=-1
			#print(h_line[90]) (giving error in some case)
			ang=(ang*180/math.pi)*sign
			f_img=imutils.rotate_bound(img,ang)
			#plt.imshow(f_img)
			#plt.show()
			print(ang)
			return f_img

	#13
	def bbox(img):
		histy = cv2.reduce(img,1, cv2.REDUCE_AVG).reshape(-1)
		histx = cv2.reduce(img,0, cv2.REDUCE_AVG).reshape(-1)
		th = 1
		H,W = img.shape[:2]
		uppers = np.array([y for y in range(H-1) if histy[y]<=th and histy[y+1]>th])
		lowers = np.array([y for y in range(H-1) if histy[y]>th and histy[y+1]<=th])

		left = np.array([y for y in range(W-1) if histx[y]<=th and histx[y+1]>th])
		right = np.array([y for y in range(W-1) if histx[y]>th and histx[y+1]<=th])

		img=img[min(uppers):max(lowers),min(left):max(right)]
		img = cv2.copyMakeBorder(img, 70, 70, 70, 70, cv2.BORDER_CONSTANT, None, value = 0)
		#plt.imshow(img)

		return img

	#14
	def isLineExist(img):
		histy = cv2.reduce(img,1, cv2.REDUCE_AVG).reshape(-1)
		th = 1
		H,W = img.shape[:2]
		uppers = np.array([y for y in range(H-1) if histy[y]<=th and histy[y+1]>th])
		lowers = np.array([y for y in range(H-1) if histy[y]>th and histy[y+1]<=th])
		print(uppers,"print uppers")
		print(lowers,"print lowers")
		if( (len(uppers)!=len(lowers)) or (len(uppers)==0) or (len(lowers)==0) ):
				return False
		return True

	#15
	def preprocessImage(test_img):
		#after preprocessing returns 3 channel image
		global is_valid
		cleaned_orig=cleanImage(test_img)
		if(isLineExist(cleaned_orig)==False):
			is_valid=False
			test_img=cv2.cvtColor( cv2.bitwise_not(cleaned_orig), cv2.COLOR_GRAY2RGB)
			return test_img

		#rotation part
		cleaned_orig=rotate(cleaned_orig)
		#bounding box part now this is our cleaner,rotated image
		cleaned_orig=bbox(cleaned_orig)
		#print("print final cleaned image")
		# print(cleaned_orig.shape)
		# plt.imshow(cleaned_orig)
		#convert to 3 channel for further processing
		test_img=cv2.cvtColor( cv2.bitwise_not(cleaned_orig), cv2.COLOR_GRAY2RGB)
		return test_img

	#16
	def lineExtraction1(test_img):
		#Line extraction work here 
		#extract line returns cleaned_orig, upperline index list , lowerline index list
		df_lines = pd.DataFrame()
		for r,rect in enumerate(new_workspace):
			
			#Cropping boxes for sending to line detection module
			#box = test_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
			global cleaned_orig
			box = test_img  
			H,W = test_img.shape[:2]
			cleaned_orig,y1s,y2s = extract_line(test_img, show=True)
			print(y1s,"upper")
			print(y2s,"lower")
			x1s = [0]*len(y1s)
			x2s = [W]*len(y1s)

			df = pd.DataFrame([y1s,y2s,x1s,x2s]).transpose()
			df.columns = ['y1','y2','x1','x2']
			df['box_num'] = r

			df_lines= pd.concat([df_lines, df])
			

			dict_clean_img.update({r:cleaned_orig})
			dict_img.update({r:box})
			#print('\n')
			#print(df_lines)

		df_lines['line_name'] = ['%d%d' %(df_lines.box_num.iloc[i],df_lines.index[i]) \
						for i in range(len(df_lines))]
		
		return df_lines


	#17
	def removeDashLines(df_lines):

		# remove useless  dash lnes
		print("before removing useless lines/rows \n",len(df_lines))
		#rows to remove
		delete_rows=[]
		for index, row in df_lines.iterrows():
				Y1=row["y1"]
				Y2=row["y2"]
				X1=row["x1"]
				X2=row["x2"]
				box_num=row["box_num"]
				line_name=row["line_name"]

				dict_clean=dict_clean_img
				img = dict_clean[box_num][Y1:Y2,X1:X2].copy()
				L_H = Y2-Y1
				## apply some dilation and erosion to join the gaps
				#Selecting elliptical element for dilation    
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
				dilation = cv2.dilate(img,kernel,iterations = 2)
				erosion = cv2.erode(dilation,kernel,iterations = 1)
				
				# Find the contours
				if(cv2.__version__ == '3.3.1'):
						xyz,contours,hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
				else:
						contours,hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
						
				ct_th = find_good_contours_thres(contours, alpha=0.005)
				cnts = []
				for c in contours:     
						if( cv2.contourArea(c)**2 > ct_th ):
								cnts.append(c)
				contours_sorted,bounding_boxes = sort_contours(cnts,method="left-to-right")

				if len(contours_sorted) == 1:
					x,y,w,h = bounding_boxes[0]
					if w > 3*h:
						delete_rows.append(index)


		print("delete rows  ",delete_rows)
		for i in delete_rows:
			df_lines=df_lines.drop(labels=i,axis=0)

		df_lines = df_lines.reset_index(drop=True)
		print("after removing useless rows \n",len(df_lines))

		for i in range(len(df_lines)) :
			line_new_name=str(0)+str(i)
			df_lines.loc[i,"line_name"]=line_new_name

		print("after reseting line name \n",df_lines)


		#plot rectangles in image
		#cleaned_orig_rec=cleaned_orig
		# cleaned_orig_rec = cv2.cvtColor(cleaned_orig, cv2.COLOR_GRAY2BGR)
		# col_ct = 0
		# for i in range(len(df_lines)):
		#   X1=df_lines.loc[i,"x1"]
		#   X2=df_lines.loc[i,"x2"]
		#   Y1=df_lines.loc[i,"y1"]
		#   Y2=df_lines.loc[i,"y2"]
			
		#   col1 = (80,150,255)
		#   col2 = (0,255,0)
		#   if(col_ct % 2 == 0):
		#     col= col1
		#   else:
		#     col=col2
		#   cv2.rectangle(cleaned_orig_rec ,(X1+15,Y1),(X2-25,Y2),col,5)
		#   col_ct += 1
			
		#cleaned_orig_rec = cv2.cvtColor(cleaned_orig_rec, cv2.COLOR_BGR2RGB)
		#plt.imshow(cleaned_orig_rec)

		# fig0 = plt.figure(figsize=(15,9))
		# ax1 = fig0.add_subplot(1,1,1)
		# ax1.set_title('final line detection')
		# ax1.imshow(cleaned_orig_rec)
		# ax1.axis('off')
		# plt.figure(figsize=(15,9))
		# plt.imshow(cleaned_orig_rec)
		return df_lines


	#18
	def lineExtraction2(list_chars):
		#list char is list of [boxno,linename, and dataframe]

		#Preprocess list char by inserting visited column and sorting them according to Y1 values for extraction of expressions line by line which is performed in next section
		#adding visited column, it will help in creating new data frame
		for lines in list_chars:
			data=lines[2]
			data['visited']=False

		# print("available  data frames ")
		# for lines in list_chars:
		#   print(lines[2])

		#print("Sort according to Y1 values")
		#sort according to Y1 values ,after sorting need to do reset indexing
		for lines in list_chars:
			data=lines[2]
			data=data.sort_values(by = 'Y1')
			lines[2]=data
			lines[2]=lines[2].reset_index(drop=True)
		
		# print("sorted dataframe")
		# for lines in list_chars:
		#   print(lines[2])

		#Extracting out expression line by line even when spaces are not enough between lines 
		#Cases may be there of many lines in one extracted line

		#list_lines is a list of  [line_upper_limit, line_below_limit, dataframe]
		list_lines=[]
		#line_num helps in updating line number across lines
		line_num=1;

		for lines in list_chars:
			#take out org dataframe line by line and make a copy so that org remain same
			org_data=lines[2]
			data=org_data.copy(deep=True)

			while 1:
				#data has current dataframe and this while loop extract all the lines that are in current dataframe and line number and line limits
				flag=False
				
				for i in range(len(data)) :
					
					if (data.loc[i,'visited'] == True):
						continue
					else:
						#find starting point or 1st box and take all boxes whose cmoy lie between these boxes
						flag=True
						#temp_list stores all information ie.. all lists ,,it will be list of data dataframe rows and after current line end we will make it dataframe and add to list_lines as a list of
						#  [line_lowert_limit , line upper limit, line]
						temp_list=[]
						l1=data.loc[i,'Y1']
						l2=data.loc[i,'Y2']
						#for slanted writing incrementing lower limit by 15 percent
						#increment=((l2-l1)*20)/100
						#l2=l2+increment
						data.loc[i,'visited']=True
						temp_list.append( [data.iloc[i,0], data.iloc[i,1], data.iloc[i,2], data.iloc[i,3], data.iloc[i,4], data.iloc[i,5], data.iloc[i,6], line_num]) 
						#move from next bcz before this all are visited
						for j in range(i+1,len(data)):
							if data.loc[j,'visited'] == True:
								continue
							else:
								u1=data.loc[j,'Y1']
								u2=data.loc[j,'Y2']
								comy=(u1+u2)/2
								if (comy>=l1 and comy<=l2):
									data.loc[j,'visited']=True
									temp_list.append([data.iloc[j,0], data.iloc[j,1], data.iloc[j,2], data.iloc[j,3], data.iloc[j,4], data.iloc[j,5], data.iloc[j,6], line_num]) 
									#after selecting this one now use limit of this one it will detect correct lines even if characters of lines are written in slanted way 
									l1=u1
									l2=u2
								elif( (u1>=l1 and u1<l2) ):
									percent= ((abs(u1-l2)/abs(u1-u2))*100)
									if(percent>=45):
										data.loc[j,'visited']=True
										temp_list.append([data.iloc[j,0], data.iloc[j,1], data.iloc[j,2], data.iloc[j,3], data.iloc[j,4], data.iloc[j,5], data.iloc[j,6], line_num]) 
										l1=u1
										l2=u2

						
						df2=pd.DataFrame(temp_list, columns=['X1' ,'Y1' ,'X2' ,'Y2' ,'area' ,'pred' ,'pred_proba' ,'line_num'])
						df2=df2.sort_values(by='X1')
						df2=df2.reset_index(drop=True)
						#line limit
						line_upper_limit=df2['Y1'].min()
						line_lower_limit=df2['Y2'].max()
						list_lines.append( [line_upper_limit, line_lower_limit, df2] )
						line_num+=1

				if flag==False:
					break
				
				
		print(len(list_lines))
		print('\n')
		for i in range(len(list_lines)):
			print(list_lines[i][0], '\n')   
			print(list_lines[i][1], '\n')   
			print(list_lines[i][2], '\n')

		# img=dict_clean_img[0].copy()
		# h,w = img.shape[:2]
		# col1=(0,0,255)
		# col2=(0,255,0)
		# img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# for i in range(len(list_lines)):
		#     Y1=list_lines[i][0]
		#     Y2=list_lines[i][1]
		#     X1=0+15
		#     X2=w-15
		#     if i%2!=0:
		#       col=col1
		#     else:
		#       col=col2
		#     cv2.rectangle(img ,(X1,Y1),(X2,Y2),col,4)
		# plt.figure(figsize=(15,9))
		# plt.imshow(img)
		return list_lines


	# Block 7

	#evaluation functions
	#1 def div(num2,num1)
	#2 def num2list(num):
	#3 def list2num(lst):
	#4 isValid(num1,num2,operator):
	#5 add(num1,num2,data,line_idx):
	#6 sub(num1,num2,data,line_idx)
	#7 mul(num1,num2,data,line_idx):
	#8 isDivision(list_lines):
	#9 def fun1(list_lines):
	#10 def evaluation(list_lines):

	#1
	def div(num2,num1):
			quo=num2//num1
			rem=num2%num1
			if(num1>num2):return[str(num1)+")"+str(num2)+"("+str(quo),str(0),str(num2)]
			else:
					quo_list=num2list(quo)
					num2_list=num2list(num2)
					ans=[]
					line1=str(num1)+")"+str(num2)+"("+str(quo)
					ans.append(line1)
					imm_num,idx=(0,0)
					while(idx<len(num2_list)):
							#print(idx,imm_num)
							while(imm_num<num1 and idx<len(num2_list)):
									#print(imm_num,idx,num2_list)
									imm_num=imm_num*10+num2_list[idx]
									idx+=1
							if(imm_num>=num1):
									ans.append(str(imm_num))
									ans.append(str(num1*(imm_num//num1)))
									imm_num=imm_num%num1
							else:idx+=1

					ans.append(imm_num)
					ans.pop(1)
					return ans
							
			
		
	#2 (used by 1)  
	def num2list(num):
			num_list=[]
			while(num!=0):
					num_list.append(num%10)
					num=(num-num%10)//10
			num_list=num_list[::-1]
			return num_list

	#3 (used by 1)
	def list2num(lst):
			num=0
			for n in range(len(lst)):
					num+=lst[-1-n]*(10**n)
			return num

	#4
	def isValid(num1,num2,operator):
		if(num1=="" or num2=="" or operator=="" ):
			return False

		if (operator!='+' and operator!='-' and operator!='*' ):
			return False

		for i in num1:
			if(i<'0' or i>'9'):
				return False

		for i in num2:
			if(i<'0' or i>'9'):
				return False

		return True

	#5
	def add(num1,num2,data,line_idx):

		wrong_lines=[]
		line_idx+=1
		ans= str(int(num1)+int(num2))

		#removing leading zeros
		value=data[line_idx]
		n=len(value)
		k=n
		for i in range(len(value)):
			if (value[i]!='0'):
				k=i
				break
		
		if(k==n):
			value='0'
		else:
			value=value[k:]

		if (value!=ans):
			Y1=list_lines[line_idx][0]
			Y2=list_lines[line_idx][1]
			wrong_lines.append([Y1,Y2,ans]) 

		return wrong_lines

	#6
	def sub(num1,num2,data,line_idx):
		wrong_lines=[]
		line_idx+=1
		ans= str(int(num1)-int(num2))

		#removing leading zeros
		value=data[line_idx]
		n=len(value)
		k=n
		for i in range(len(value)):
			if (value[i]!='0'):
				k=i
				break
		
		if(k==n):
			value='0'
		else:
			value=value[k:]

		if (value!=str(ans)):
			Y1=list_lines[line_idx][0]
			Y2=list_lines[line_idx][1]
			wrong_lines.append([Y1,Y2,ans]) 

		return wrong_lines

	#7
	def mul(num1,num2,data,line_idx):
		wrong_lines=[]
		rev_num2=num2[::-1]
		digits=len(rev_num2)
		n=len(rev_num2)


		#intermediate lines
		for i in range(n):
			
			n1=rev_num2[i]
			temp= str(int(num1)*int(n1))
			line_idx+=1
			
			#removing leading zeros
			value=data[line_idx]
			m=len(value)
			k=m
			for j in range(len(value)):
				if (value[j]!='0'):
					k=j
					break

			if(k==m):
				value='0'
			else:
				value=value[k:]
				if(len(value)==1 and value=='*'):
					value='0'+value
			
			if(i==0):
				#1st intermediate line

				if(temp!=value):
					Y1=list_lines[line_idx][0]
					Y2=list_lines[line_idx][1]
					wrong_lines.append([Y1,Y2,temp])
				
			elif(i==n-1):
				
				#last intermediate line and is is special case bcz operator may  be added at front so multiple cases are possible

				temp=str(int(num1)*int(n1))
				temp2=temp
				for j in range(i):
					temp2+='0'

				temp1=temp+'*'
				temp3='+' + temp1
				temp4='+' + temp2
				if(temp1!=value and temp2!=value and temp3!=value and temp4!=value ):
					Y1=list_lines[line_idx][0]
					Y2=list_lines[line_idx][1]
					wrong_lines.append([Y1,Y2,temp])

			else:
				#these lines are neither 1st nor last intermediate lines

				temp=str(int(num1)*int(n1))
				temp2=temp
				for j in range(i):
					temp2+='0'

				temp1=temp+'*'
				
				if(temp1!=value and temp2!=value ):
					print(temp1 ,"  ",temp2,"  ",value)
					Y1=list_lines[line_idx][0]
					Y2=list_lines[line_idx][1]
					wrong_lines.append([Y1,Y2,temp])


		#Intermediate lines cheked ,, now check last lines which has main answer and check only if n>1
		if(n>1):
			line_idx+=1
			ans= str(int(num1)* int(num2))

			#removing leading zeros
			value=data[line_idx]
			n=len(value)
			k=m
			for j in range(len(value)):
				if (value[j]!='0'):
					k=j
					break
			
			if(k==m):
				value='0'
			else:
				value=value[k:]
			if(ans!=value):
				Y1=list_lines[line_idx][0]
				Y2=list_lines[line_idx][1]
				wrong_lines.append([Y1,Y2,ans])

		return wrong_lines


	#8
	def isDivision(list_lines):
		#output true if division question else false for addition , multiplication and subtraction
		parenthesis=[]
		
		for j in range(len( list_lines[0][2]["pred"] )):
			ch=list_lines[0][2]["pred"][j]
			if(ch=='(' or ch==')'):
				parenthesis.append(ch)

		if(len(parenthesis)==2 and parenthesis[0]==')' and parenthesis[1]=='('):
			return True
		else:
			return False  


	#9
	def fun1(list_lines):
		#take list_lines data frame and return list of strings where each
		#string represents one line 
		data = []
		expr=""
		for i in range(len(list_lines)):
			for j in range(len( list_lines[i][2]["pred"] )):
				expr+=list_lines[i][2]["pred"][j]
			data.append(expr)
			expr=""

		print("line by line representation of solved question ")
		print("-----------------------")
		for i in range(len(data)):
			print(data[i],'\n')
		print("-----------------------")

		return data


	#10
	def evaluation(list_lines):
		#evaluate, mark valid/invalid,num1,num2,operator and wrong lines
		#wrong lines is list of [upper, lower, correct]
		#stores list of  (wrong lines) [Y1,Y2,correct value]

		global num1
		global num2
		global operator
		global is_valid

		wrong_lines=[]
		if(len(list_lines)==0 or len(list_lines)==1  ):
			is_valid=False
			return wrong_lines

		else:
			is_division=isDivision(list_lines)
			#data stores lines as character strings
			data = fun1(list_lines)

			if(is_division==False):
				print("is_division",is_division)
				#addition, subtraction or multiplication operation is performed

				#check question is in single line or multiline
				in_single_line=False
				for idx in range( len(data[0] )):
					if (data[0][idx]=='*' or data[0][idx]=='+' or data[0][idx]=='-' ):
						in_single_line=True
						break

				#line_idx points to data list which stores our image line by line as string
				#it tells this line_idx evaluation is completed
				line_idx=0
				print("in_single_line",in_single_line)

				if(in_single_line):
					#question is written in single line
					expr=""
					#separate num1, num2 and operator
					for idx in range( len(data[0] )):
						if (data[0][idx]!='*' and data[0][idx]!='+' and data[0][idx]!='-' ):
							expr+=data[0][idx]
						else:
							operator=data[0][idx]
							num1=expr
							expr=""
							j=idx+1
							while j!=( len(data[0] )):
								expr+=data[0][j]
								if(j==len(data[0])-1):
									num2=expr 
								j+=1
					


					line_idx=0

				else:
					#question is written in two line so atleast 2 lines should be present 
					#if only one line present then invalid format
					
					if(len(data)<2):
						num1=data[0]
						is_valid=False
						return wrong_lines

					num1= data[0]
					
					temp=data[1][0:1]
					if(temp!='+' and temp!='-' and temp!='*'):
						operator=""
						num2= data[0]
					else:
						operator= data[1][0:1]
						num2= data[1][1:]

					line_idx=1

				print("num1 = ",num1 ,'\n')
				print("operator = ",operator ,'\n')
				print("num2 = ",num2 ,'\n')

				#cheking equation valid or not
				is_valid=isValid(num1,num2,operator)
		
				if (is_valid==False):
					print("Invalid Numbers Cant Evaluate")
					return wrong_lines
				else:
					#evaluation as num1,num2,operator are initialized
					if (operator=='+'):
						#case when operator is +
						if((in_single_line==True) and len(data)!=2):
							is_valid=False
							return wrong_lines
						elif((in_single_line==False) and len(data)!=3):
							is_valid=False
							return wrong_lines

						#pass num1,num2,data(list of all lines),line_idx (last line of question or line before solution started)
						list_wrong_lines=add(num1,num2,data,line_idx)
						for j in list_wrong_lines:
							wrong_lines.append(j)
						
					elif (operator=='-'):
						#case when operator is -

						if((in_single_line==True) and len(data)!=2):
							is_valid=False
							return wrong_lines
						elif((in_single_line==False) and len(data)!=3):
							is_valid=False
							return wrong_lines

						#pass num1,num2,data(list of all lines),line_idx (last line of question or line before solution started)
						list_wrong_lines=sub(num1,num2,data,line_idx)
						for j in list_wrong_lines:
							wrong_lines.append(j)
							
					else:
						#case when operator is *
						#pass num1,num2,data(list of all lines),line_idx (last line of question or line before solution started)

						num2_len=len(num2)

						if((in_single_line==True) and num2_len==1 and len(data)!=num2_len+1):
							is_valid=False
							return wrong_lines
						elif((in_single_line==True) and num2_len>1 and len(data)!=num2_len+2):
							is_valid=False
							return wrong_lines
						elif((in_single_line==False) and num2_len==1 and len(data)!=num2_len+2):
							is_valid=False
							return wrong_lines
						elif((in_single_line==False) and num2_len>1 and len(data)!=num2_len+3):
							is_valid=False
							return wrong_lines

						list_wrong_lines=mul(num1,num2,data,line_idx)
						for j in list_wrong_lines:
							wrong_lines.append(j)

				return wrong_lines
			else:
				#division operation is performed
				
				print("is_division",is_division)

				#collecting num1 and num2
				l_id=data[0].index(')')
				r_id=data[0].index('(')
				num1=data[0][0:l_id]
				num2=data[0][l_id+1:r_id]

				#cheking validity of num1 and num2
				for i in num1:
					if(i<'0' or i>'9'):
						is_valid=False
						break

				for i in num2:
					if(i<'0' or i>'9'):
						is_valid=False
						break

				if (is_valid==False):
					print("Invalid Numbers Cant Evaluate \n")
					is_valid=False
					return wrong_lines
				else:
					print("num1 ",num1)
					print("num2 ",num2)

					correct_lines=div(int(num2),int(num1))
					#give correct lines if solve but not contain -sign in odd idx lines ,so we have to add
					if(len(correct_lines)!=len(data)):
						is_valid=False
						return wrong_lines

					for i in range(len(data)):
						if(data[i][0]=='-'): data[i]=data[i][1:]

					

					if(data[0]!=correct_lines[0]):
						Y1=list_lines[0][0]
						Y2=list_lines[0][1]
						wrong_lines.append([Y1,Y2,correct_lines[0]]) 

					for i in range(len(data))[1:]:
						if(int(data[i])!=int(correct_lines[i])):
							Y1=list_lines[i][0]
							Y2=list_lines[i][1]
							wrong_lines.append([Y1,Y2,correct_lines[i]]) 

				
				return wrong_lines

	# Block 8

	#step 1 image reading
	print("\n---------------step 1 image reading---------------\n")
	org_img=cv2.imread("upload/input.jpg")

	# plt.figure(figsize=(15,20))
	# plt.subplot(1, 2, 1)
	# plt.title("Original Image")
	# plt.imshow(org_img)
	# image_gray2 = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
	# thresh, binary_image2 = cv2.threshold(image_gray2, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# plt.subplot(1, 2, 2)
	# plt.title("Otsu Thresholding")
	# plt.imshow(cv2.cvtColor(binary_image2, cv2.COLOR_BGR2RGB))

	#step 2 shadow removal
	print("\n---------------step 2 shadow removal---------------\n")
	test_img=shadowRemoval(org_img)


	#getting coordinates of shadowless image
	h,w,c= test_img.shape
	a,b,c,d=0,0,w,h
	rectangle_locs =[]
	rectangle_locs.append([a,b,c,d])   
	new_workspace=rectangle_locs

	#step 3 image preprocessing ,validity cheking ,(cleaning,rotation,boundingbox)
	print("\n---------------step 3 image preprocessing ,validity cheking ,(cleaning,rotation,boundingbox)---------------\n")
	test_img=preprocessImage(test_img)
	if(is_valid==False):
		output_img=invalidImage()
		print("invalid image exit")
		return


	#step 4 line extraction 1
	print("\n---------------step 4 line extraction 1---------------\n")
	df_lines=lineExtraction1(test_img)
	print(df_lines)

	#step 5 remove Dash lines
	print("\n---------------step 5 remove Dash lines---------------\n")
	df_lines=removeDashLines(df_lines)

	#step 6 extraction of characters (text segmentation)
	print("\n---------------step 6 extraction of characters (text segmentation)---------------\n")
	list_chars = list(df_lines.apply(lambda row: text_segment(row['y1'],row['y2'],row['x1'],row['x2'], row['box_num'],row['line_name'],show=True), axis=1))

	#What we get after performing textsegment in each line
	#text segemnt return [box_num,line_name,df_char (char information of each line)]
	print(np.shape(list_chars))

	for i in range(len(list_chars)):
		print(list_chars[i][0], '\n')   
		print(list_chars[i][1], '\n')   
		print(list_chars[i][2], '\n')  

	#step 7 line extraction 2
	print("\n---------------step 7 line extraction 2---------------\n")
	list_lines=lineExtraction2(list_chars)

	#step 8 evaluation,validitiy cheking ,get wrong lines info
	print("\n---------------step 8 evaluation,validitiy cheking ,get wrong lines info---------------\n")
	wrong_lines=evaluation(list_lines)
	print(is_valid,"is_valid")
	if(is_valid==False):
		output_img=invalidImage()
		return
	print(wrong_lines)

	#step 9 output image
	print("\n---------------step 9 output image---------------\n")
	if is_valid==True:
		output_img = test_img.copy()
		
		h,w,c = output_img.shape

		if(len(wrong_lines)==0):
			output_img=correctImage(output_img)
		else:  
			output_img=wrongImage(output_img,wrong_lines)

		output_img = cv2.putText(output_img, num1+" "+operator+" "+num2 , (15,30),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230,100,255), 3, cv2.LINE_AA)
		# plt.figure(figsize=(12,12))
		cv2.imwrite("./upload/final.png",output_img)
