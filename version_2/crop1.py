import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


#this is other way to crop paper using hough transform but it is very unstable
#so i am not commenting it

#1
def polygon(lst):
    a=lst[0]
    ang_diff=[abs(lst[0][1]-lst[1][1]),abs(lst[0][1]-lst[2][1]),abs(lst[0][1]-lst[3][1])]
    #print(ang_diff)
    idx=1+ang_diff.index(min(ang_diff))
    #print(idx)
    c=lst[idx]
    b_d_idx=[1,2,3]
    #print(b_d_idx)
    b_d_idx.remove(idx)
    #print(b_d_idx)
    b,d=(lst[b_d_idx[0]],lst[b_d_idx[1]])
    #print(a,c)
    #print(b,d)
    A=line([a[0]*math.sin(a[1]),a[0]*math.cos(a[1])],[a[0]*math.sin(a[1])+10*math.cos(a[1]),a[0]*math.cos(a[1])-10*math.sin(a[1])])
    B=line([b[0]*math.sin(b[1]),b[0]*math.cos(b[1])],[b[0]*math.sin(b[1])+10*math.cos(b[1]),b[0]*math.cos(b[1])-10*math.sin(b[1])])
    C=line([c[0]*math.sin(c[1]),c[0]*math.cos(c[1])],[c[0]*math.sin(c[1])+10*math.cos(c[1]),c[0]*math.cos(c[1])-10*math.sin(c[1])])
    D=line([d[0]*math.sin(d[1]),d[0]*math.cos(d[1])],[d[0]*math.sin(d[1])+10*math.cos(d[1]),d[0]*math.cos(d[1])-10*math.sin(d[1])])
    #print("hai",[a[0]*math.sin(a[1]),a[0]*math.cos(a[1])],[a[0]*math.sin(a[1])+10*math.cos(a[1]),a[0]*math.cos(a[1])-10*math.sin(a[1])])
    #print([b[0]*math.sin(b[1]),b[0]*math.cos(b[1])],[b[0]*math.sin(b[1])+10*math.cos(b[1]),b[0]*math.cos(b[1])-10*math.sin(b[1])])
    #print([c[0]*math.sin(c[1]),c[0]*math.cos(c[1])],[c[0]*math.sin(c[1])+10*math.cos(c[1]),c[0]*math.cos(c[1])-10*math.sin(c[1])])
    #print([d[0]*math.sin(d[1]),d[0]*math.cos(d[1])],[d[0]*math.sin(d[1])+10*math.cos(d[1]),d[0]*math.cos(d[1])-10*math.sin(d[1])])
    ans=[intersection(A,B),intersection(B,C),intersection(C,D),intersection(D,A)]
    #print(ans,"ans")
    ans1=[sum(intersection(A,B)),sum(intersection(B,C)),sum(intersection(C,D)),sum(intersection(D,A))]
    u_l,l_r=(ans[ans1.index(min(ans1))],ans[ans1.index(max(ans1))])
    ans.remove(u_l)
    ans.remove(l_r)
    #print(ans,"jjjj")
    if(ans[0][0]>ans[1][0]):
        u_r,l_l=ans
    else:
        u_r,l_l=ans[::-1]
    #print(ans1)
    ans=[u_l,u_r,l_r,l_l]
    return ans

#2       
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

#3
def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return y,x
    else:
        return False
    
#4
def hough_lines(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    size=img.shape
    #print(size,"hjy")
    lines=cv2.HoughLines(img,1,np.pi/180,200)
    lst=[]
    #print(lines,"ute")
    for r_theta in lines:
        #print(r_theta)
        #print(r_theta)
        r,theta = r_theta[0]
        diff=1000
        for n in lst:
            diff=min(diff,abs(n[0]-r)+200*abs(n[1]-theta))
        #print(diff)             
        if(diff>200):
            lst.append((r,theta))
    return lst


#5   
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
    #print(center)
    res=center[label.flatten()]
    result_image=res.reshape((temp.shape))
    th,binary_mask=cv2.threshold(result_image,125,255,cv2.THRESH_BINARY)
    binary_mask=cv2.erode(binary_mask, np.ones((75,75), np.uint8))
    return (binary_mask,cv2.mean(img,binary_mask[:,:,0]))


#6
def crop(img):
    print(img.shape,"uu")
    mask,label=segment_paper(img[50:-50,50:-50])
    #print(label,"label")
    #print(mask.shape,"ll")
    mask_extended=cv2.copyMakeBorder(mask, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0])
    edge=mask_extended-cv2.erode(mask_extended,np.ones((7,7), np.uint8))
    #plt.imshow(edge,cmap="gray")
    #plt.show()
    """
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(mask_extended[i,j,0]==0): 
                img[i,j]=(label[0],label[1],label[2])"""
    r,g,b=(int(label[0]),int(label[1]),int(label[2]))
    img=cv2.bitwise_and(img,mask_extended)
    img2=img+(mask_extended==0)*(r,g,b)
    img1=(mask_extended==0)*(r,g,b) 
    img2=np.array(img2,dtype="uint8")
    #img=cv2.bitwise_or(img,255-mask_extended)
    #plt.imshow(img,cmap="gray")
    #plt.show()
    #print(edge.shape,"ll")
    lines=hough_lines(edge)
    #print(lines,"lines")
    if(len(lines)==4):
        #rows,cols = img.shape
        pts1 = np.float32(polygon(lines))#([[359,359],[1259,375],[1179,1885],[226,1869]])
        pts2 = np.float32([[0,0],[300,0],[300,450],[0,450]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        img = cv2.warpPerspective(img,M,(300,450))
        #plt.subplot(121),plt.imshow(img,cmap="gray"),plt.title('Input')
        #plt.subplot(122),plt.imshow(img,cmap="gray"),plt.title('Output')
        #plt.show()
    return (img2,img1,mask_extended)