from sklearn.cluster import DBSCAN
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

import paper

#1
def problems(img):
    """
    function to divide image with multiple problems into images of subproblems
    input:image
    output:list of images of problems,list of bounding boxes of problems
    """
    clusters,images=({},[])
    img=img.copy()
    img=paper.filter_img2_chr(img)#filter the image
    contours,hierarchy = cv2.findContours(img[0:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=[i for i in contours if(len(i)>0)]
    boundingBoxes = [cv2.boundingRect(c) for c in contours]#find bounding boxes
    bb=[]
    for n in range(len(boundingBoxes)):
        bb.append(list(c_o_mass(boundingBoxes[n])))
    bb=np.array(bb)
    dbscan_cluster1=DBSCAN(eps=300,min_samples=2)
    dbscan_cluster1.fit(bb)
    labels=dbscan_cluster1.labels_
    for n in range(len(labels)):
        if(labels[n] not in clusters):
            clusters.update({labels[n]:[boundingBoxes[n]]})
        else:
            clusters[labels[n]].append(boundingBoxes[n])
    for n in clusters:
        images.append(convert_to_image(img,clusters[n]))
    images=sorted(images,key=lambda x:x[1][1])
    return images
             
    
    
#2 function to return center of bounding box   
def c_o_mass(box):
    x,y,l,b=box
    return (x+(l//2),y+(b//2))


#3
def convert_to_image(img,lst):
    """
    function to extract image of problem from list of bounding boxes
    input:image,list of bounding boxes
    output:image,bounding box of problem in the image
    """
    up=sorted(lst,key=lambda x:x[1])[0][1]
    left=sorted(lst,key=lambda x:x[0])[0][0]
    down=sorted(lst,key=lambda x:x[1]+x[3],reverse=True)[0][1]+sorted(lst,key=lambda x:x[1]+x[3],reverse=True)[0][3]
    right=sorted(lst,key=lambda x:x[0]+x[2],reverse=True)[0][0]+sorted(lst,key=lambda x:x[0]+x[2],reverse=True)[0][2]
    return (img[up:down,left:right],(left,up,right-left,down-up))
    


