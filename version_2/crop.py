import math
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

#1 function to find p1p2+p2p3 (p1p2 is distance between point p1 and point p2)
def dist(p1,p2,p3):
    p1p2=math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    p2p3=math.sqrt((p2[0]-p3[0])**2+(p2[1]-p3[1])**2)
    return p1p2+p2p3


#2 function to find corners of the paper
def find_paper_corners(img):
    img=img.copy()
    corners=[]
    img,mean =segment_paper(img)
    if(cv2.__version__ == '3.3.1'):
          xyz,contours,hierarchy = cv2.findContours(img[:,:,0],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    else:
          contours,hierarchy = cv2.findContours(img[:,:,0],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    mask=np.zeros(img.shape[0:2])
    cv2.drawContours(mask,contours,-1,255,1)
    mid_points_indices,x_half,y_half=([],img.shape[0]//2,img.shape[1]//2)
    for n in range(len(contours[0])):
        if(contours[0][n][0][0]==y_half or contours[0][n][0][1]==x_half):
            mid_points_indices.append(n)
    for i in range(4):
        mid, nxt_mid, idx, dst=(mid_points_indices[i], mid_points_indices[(i+1)%4], 0, 0) 
        for n in range(mid_points_indices[i],mid_points_indices[(i+1)%4]):
            if dist(contours[0][mid][0],contours[0][n][0],contours[0][nxt_mid][0])>dst:
                idx,dst=n,dist(contours[0][mid][0],contours[0][n][0],contours[0][nxt_mid][0])
        corners.append(contours[0][idx][0])
    up_cor,low_cor=(sorted(corners,key=lambda x:x[1])[0:2],sorted(corners,key=lambda x:x[1])[2:4])
    ul,ur=sorted(up_cor,key=lambda x:x[0])
    ll,lr=sorted(low_cor,key=lambda x:x[0])
    return [ul,ur,lr,ll]


#3 function to plot corners of paper in the image
def plot_corners(img):
    img=img.copy()
    corners=find_paper_corners(img)
    img,mean =segment_paper(img)
    print(corners)
    plt.imshow(img)
    plt.plot(corners[0][0],corners[0][1],marker="v")
    plt.plot(corners[1][0],corners[1][1],marker="v")
    plt.plot(corners[2][0],corners[2][1],marker="v")
    plt.plot(corners[3][0],corners[3][1],marker="v")
    plt.show()
    return None
 
    
#4 function to segment the paper from background   
def segment_paper(img):
  
    temp=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    med_fil_img=cv2.medianBlur(temp,55)
    rgb_img=cv2.cvtColor(med_fil_img,cv2.COLOR_BGR2RGB)
    two_d=rgb_img.reshape((-1,3))
    two_d=np.float32(two_d)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    k=2
    attempts=10
    compactness,label,center=cv2.kmeans(two_d,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    
    center=np.uint8(center)
    res=center[label.flatten()]
    result_image=res.reshape((temp.shape))
    th,binary_mask=cv2.threshold(result_image,125,255,cv2.THRESH_BINARY)
    binary_mask=cv2.erode(binary_mask, np.ones((75,75), np.uint8))
    return (binary_mask,cv2.mean(img,binary_mask[:,:,0]))


#function to remove background and return image that is completely filled with paper
def remove_background(img):
    orig_img=img
    img=orig_img.copy()
    w,l=(img.shape[0],img.shape[1])
    mask,label=segment_paper(img[50:-50,50:-50])
    mask_extended=cv2.copyMakeBorder(mask, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0])
    
    r,g,b=(int(label[0]),int(label[1]),int(label[2]))
    img=cv2.bitwise_and(img,mask_extended)
    img=img+(mask_extended==0)*(r,g,b)
    img=np.array(img,dtype="uint8")
    
    pts1 = np.float32(find_paper_corners(orig_img))
    pts2 = np.float32([[0,0],[l,0],[l,w],[0,w]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    img = cv2.warpPerspective(img,M,(l,w))
    
    return img
 