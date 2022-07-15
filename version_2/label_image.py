import cv2
import numpy as np
###########################################

#image labelling functions

#1
def invalidImage(img):
    """
    invalidImage:function to label "Invalid" on top right of "img"
    argument:
        img(array)  : image
    output:
        output_img(array) : image with extra 60 rows with label "Invalid" on topright
    """
    img=cv2.copyMakeBorder(img,60,0,0,0,cv2.BORDER_CONSTANT,value=0)#adding 60 balnk rows at top
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_x = img.shape[1]-320
    bottom_left_y = 60
    bottom_left = ( bottom_left_x , bottom_left_y ) 

    fontScale,color,thickness = (1.2, (230,100,255), 3)#parameters to specify type of text
    output_img = cv2.putText(img, "Invalid" , bottom_left, font,fontScale, color, thickness, cv2.LINE_AA)
    return output_img
    

    
#2
def correctImage(img):
    """
    correctImage:function to label "Correctly Solved" on top right of "img"
    argument:
        img(array)  : image
    output:
        output_img(array) : image with extra 60 rows with label "Correctly Solved" on topright
    """
    img=cv2.copyMakeBorder(img,60,0,0,0,cv2.BORDER_CONSTANT,value=0)#adding 60 balnk rows at top
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_x = img.shape[1]-320
    bottom_left_y = 60
    bottom_left = ( bottom_left_x , bottom_left_y ) 

    fontScale,color,thickness = (1.2, (230,100,255), 3)#parameters to specify type of text
    output_img = cv2.putText(img, "Correctly Solved" , bottom_left, font,fontScale, color, thickness, cv2.LINE_AA)
    return output_img



#3
def wrongImage(img,wrong_lines):
    """
    wrongImage:function to label "wrongly Solved" on top right of "img" and label the wrong lines in the img with correct answer
    argument:
        img(array)                                   : image
        wrong_lines(list(y1(int),y2(int),ans(str)))  : y1,y2 left and right margins of wrong line,ans=correct answer at that line
    output:
        output_img(array) : image with extra 60 rows with label "wrongly Solved" on topright and wrong lines labelled with correct ans
    """
    col1,col2=((0,0,255),(255,0,0))
    for i in range(len(wrong_lines)):
        Y1,Y2,correct_ans=(wrong_lines[i][0], wrong_lines[i][1], wrong_lines[i][2])# Y-boundries of wrong line and expected correct ans at    that line
    X1,X2=(0+15,img.shape[1]-15)#left and right margins of wrong line
    col=col1
    if i%2!=1:col=col2#change colour of alternate wrong lines
    cv2.rectangle(img ,(X1,Y1),(X2,Y2),col,4)

    #write correct value in rectangular box
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_x = int( X2 - (X2-X1)/4 )
    bottom_left_y = int(Y1+35)
    bottom_left = ( bottom_left_x , bottom_left_y ) 

    fontScale,color,thickness = (1.2, (230,100,255), 3)#parameters to specify type of text
    img = cv2.putText(img, correct_ans , bottom_left, font,fontScale, color, thickness, cv2.LINE_AA)
    ###################adding 60 blank rows at the top
    img=cv2.copyMakeBorder(img,60,0,0,0,cv2.BORDER_CONSTANT,value=0)
    bottom_left_x = img.shape[1]-320
    bottom_left_y = 60
    bottom_left = ( bottom_left_x , bottom_left_y )
    output_img = cv2.putText(img, "Wrongly Solved" , bottom_left, font,fontScale, color, thickness, cv2.LINE_AA)
    return output_img




#4
def combine_images(img_lst):
    """
    combine_images:function to concatenate list of images vertically
    argument:
    img_lst(list(imgs))=list of images to be concatenated vertically
    output=single concatenated image whith white rows in between individual images
    """
    lst,mx_x=([],sorted(img_lst,key=lambda x:x.shape[1],reverse=True)[0].shape[1])#mx_x is maximum no of columns of img in list of images
    #extending columns of each image in list to be = mx_x
    for n in range(len(img_lst)):
        imm=cv2.copyMakeBorder(img_lst[n],0,0,0,mx_x-img_lst[n].shape[1],cv2.BORDER_CONSTANT,value=0)
        imm=cv2.copyMakeBorder(imm,10,10,20,20,cv2.BORDER_CONSTANT,value=255)#add boarder around each image
        lst.append(imm)
    return cv2.vconcat(lst)#return vertically concatenated image