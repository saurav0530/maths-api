import cv2

import predict
###################################

#1
def extract_lines(img):
    """
    extract_lines:function to ectract lines of bounding_boxes of connected components
    input:binary image
    output:list of list bounding boxes of connected components
    """
    contours,hierarchy = cv2.findContours(img[0:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    
    ##############################
    #function to return center of rectangular box
    def c_o_mass(box):
        x,y,l,b=box
        return (x+(l//2),y+(b//2))
    
    ##############################
    #function to find left part of bounding boxes in 1stline given the topmost boundingbox
    def left_line(boundingBoxes_list,top):
        left,curr=([],top)
        y_margin=(curr[1],curr[1]+curr[3])
        imm=[i for i in boundingBoxes_list if c_o_mass(i)[1]>y_margin[0] and c_o_mass(i)[1]<y_margin[1] and c_o_mass(curr)[0]-c_o_mass(i)[0]>0]
        while(len(imm)>0):
            imm1=sorted(imm,key=lambda x:x[0],reverse=True)
            curr=imm1[0]
            left.append(curr)
            y_margin=(curr[1],curr[1]+curr[3])
            imm=[i for i in boundingBoxes_list if c_o_mass(i)[1]>y_margin[0] and c_o_mass(i)[1]<y_margin[1] and c_o_mass(curr)[0]-c_o_mass(i)[0]>0]
        return left[::-1]
    
    ################################
    #function to find right part of bounding boxes in 1st line given the topmost bounding box
    def right_line(boundingBoxes_list,top):
        right,curr=([],top)
        y_margin=(curr[1],curr[1]+curr[3])
        imm=[i for i in boundingBoxes_list if c_o_mass(i)[1]>y_margin[0] and c_o_mass(i)[1]<y_margin[1] and c_o_mass(curr)[0]-c_o_mass(i)[0]<0]
        while(len(imm)>0):
            imm1=sorted(imm,key=lambda x:x[0])
            curr=imm1[0]
            right.append(curr)
            y_margin=(curr[1],curr[1]+curr[3])
            imm=[i for i in boundingBoxes_list if c_o_mass(i)[1]>y_margin[0] and c_o_mass(i)[1]<y_margin[1] and c_o_mass(curr)[0]-c_o_mass(i)[0]<0]
        return right
    
    ##################################
    #function to find topmost bounding box
    def topmost(boundingBoxes_list):
         return sorted(boundingBoxes_list,key=lambda x:x[1])[0]
        
    #################################
    lines,boundingBoxes_list=([],list(boundingBoxes))
    while(len(boundingBoxes_list)>0):
        top=topmost(boundingBoxes_list)#find top most box
        left=left_line(boundingBoxes_list,top)#get the left part of topmost box
        right=right_line(boundingBoxes_list,top)#get the right part of topmost box
        line=left+[top]+right
        lines.append(line)
        for n in line:#remove the boxes from the recently detected line to get next line
            if(n in boundingBoxes_list):
                boundingBoxes_list.remove(n)
    lines=[[box for box in line if box[2]//box[3]<5 or box[2]<200] for line in lines]#remove long horizontal lines
    lines=[i for i in lines if len(i)>0]#remove long horizontal lines
    return lines




#2
def extract_characters(lines,img):
    """
    extract_characters:function to covert list of bounding boxes to list of characters
    input: (list of list of bounding boxes) and image
    output:(list of list characters) and (list of ymargin of lines)
    """
    dic={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'+',11:'-',12:'*',13:'(',14:')'}
    chr_lines=[]
    for line in lines:
        chr_line=[]
        for box in line:
            cr=dic[predict.predict(img,box[0],box[1],box[0]+box[2],box[1]+box[3])]#preditct chr in the box at the img
            chr_line.append(cr)
        chr_lines.append(chr_line)
    return (chr_lines,y_values(lines))




#3
def y_values(s_bb):
    """
    y_values:function to find y_margins of list of list of boxes
    input:list of list of boxes
    output:list of (y1,y2)
    """
    ans=[]
    for n in s_bb:
        up=sorted(n,key=lambda x:x[1])[0][1]
        down=sorted(n,key=lambda x:x[1]+x[3],reverse=True)[0][1]+sorted(n,key=lambda x:x[1]+x[3],reverse=True)[0][3]
        ans.append((up,down))
    return ans  
    