import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import imutils

#1
def remove_shadow(img):
    """
    function to remove shadows
    input :image
    output:image
    """
    rgb_planes = cv2.split(img)
    result_planes,result_norm_planes = ([], [])
    
    for plane in rgb_planes:
        plane = cv2.normalize(plane,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        dilated_img = cv2.dilate(plane, np.ones((17,17), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 25)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    
    return result_norm


#2
def filter_img2_chr(img):
    """
    function to filter the image
    input:image
    output:binary image
    """
    gray=remove_shadow(img)#removing shadows from image
    gray=cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    threshed = 255-cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,199,45)   #             
    #Selecting elliptical element for dilation    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilation = cv2.dilate(threshed,kernel,iterations = 1)
    temp = dilation.copy()
    alpha,conn_thresh = (0.03, 0)
    
    #////////////////////////////////////////////////////////////////////////////////
    #Creating a mask of only ones    
    mask = np.zeros(dilation.shape[:2], dtype="uint8") 
    conn_comp=cv2.connectedComponentsWithStats(dilation,connectivity=8)
    if(len(conn_comp[2])>1):conn_thresh=sorted(conn_comp[2][:,4],reverse=True)[1]
    #Drawing those contours which are noises and then taking bitwise and
    big_comp={}
    for id, c in enumerate(conn_comp[2]):
         if(c[4]>alpha*conn_thresh and id!=0):
            big_comp.update({id:c})
            mask1=mask[c[1]:c[1]+c[3],c[0]:c[0]+c[2]]>np.zeros([c[3],c[2]])
            mask2=conn_comp[1][c[1]:c[1]+c[3],c[0]:c[0]+c[2]]==np.ones([c[3],c[2]])*id
            mask[c[1]:c[1]+c[3],c[0]:c[0]+c[2]] =(mask1|mask2)*255
   
    return mask


#3
def rotate(filtered_img):
    """
    function to rotate the image
    input:binary image
    output:binary image
    """
    img=filtered_img.copy()
    if(cv2.__version__ == '3.3.1'):
            xyz,contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    diag=([math.sqrt(l**2+b**2) for x,y,l,b in boundingBoxes])
    mx =max(diag)
    mx_ind =diag.index(mx)
    x,y,l,b=boundingBoxes[mx_ind]
    ang, h_line, sign=(math.atan(-b/l), contours[mx_ind], 1)
    left_point_index=list(h_line[:,0,0]).index(min(h_line[:,0,0]))
    if(h_line[left_point_index][0][1]-y>b//3):sign=-1
    ang=(ang*180/math.pi)*sign
    rotation_corrected_img=imutils.rotate_bound(img,ang)
    
    return rotation_corrected_img


#4
def bbox(img):
    """
    function to remove blank space in image
    input:binary image
    output:binary image
    """
    histy = cv2.reduce(img,1, cv2.REDUCE_AVG).reshape(-1)
    histx = cv2.reduce(img,0, cv2.REDUCE_AVG).reshape(-1)
    th = 1
    H,W = img.shape[:2]
    uppers = np.array([y for y in range(H-1) if histy[y]<=th and histy[y+1]>th])
    lowers = np.array([y for y in range(H-1) if histy[y]>th and histy[y+1]<=th])

    left = np.array([x for x in range(W-1) if histx[x]<=th and histx[x+1]>th])
    right = np.array([x for x in range(W-1) if histx[x]>th and histx[x+1]<=th])

    img=img[min(uppers):max(lowers),min(left):max(right)]
    img = cv2.copyMakeBorder(img, 70, 70, 70, 70, cv2.BORDER_CONSTANT, None, value = 0)
    
    return img