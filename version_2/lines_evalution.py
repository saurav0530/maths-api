import copy
#function to evaluate lines of characters in a problem 
def evaluate(lines,y_margin):
    
    list_lines =y_margin#[[0,1],[2,3],[4,5],[6,7],[8,9]]
    dic={"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0,"+":0,"-":0,"*":0,")":0,"(":0}
    valid=False
    for line in lines:
        for chr in line:
            dic[chr]+=1
      #########################function to find if only one operand in the problem
    def only_one_op(lines,dic):
        dic=copy.deepcopy(dic)
        conditions=[dic["*"]*dic["-"]>=1, dic["+"]*dic[")"]>=1, dic[")"]!=dic["("], dic[")"]*dic["("]>1, dic["-"]*dic["+"]>0,        dic[")"]*dic["*"]>0]
        print(conditions,"conditions")
        for cond in conditions:
            if(cond==True):return False
        return True
      ##########################function to classify the problem
    def find_op(lines,dic):
        if(dic["*"]>0):return "mul"
        elif(dic[")"]>0):return "div"
        elif(dic["+"]>0):return "add"
        elif(dic["-"]>0):return "sub"
      #############################function to check if syntax of addition is true
    def is_valid_add(lines,dic):
        dic=copy.deepcopy(dic)
        frst_chr=[line[0] for line in lines]
        for chr in frst_chr:
            if(chr=="+"):dic[chr]-=1
        if(dic["+"]>0):return False
        return True
      ###############################function to check if syntax of subtraction is true
    def is_valid_sub(lines,dic):
        if(len(lines)!=3):return False
        dic=copy.deepcopy(dic)
        frst_chr=[line[0] for line in lines]
        for chr in frst_chr:
            if(chr=="-"):dic[chr]-=1
        if(dic["-"]>0):return False
        return True
      #################################function to check if syntax of multiplication is true
    def is_valid_mul(lines,dic):
        lines1=copy.deepcopy(lines)
        if("*" not in lines[0] and "*" not in lines[1]):return False
        return True
      #################################function to check if syntax of division is true 
    def is_valid_div(lines,dic):
        if(")" not in lines[0] or "(" not in lines[0]):return False
        for line in lines[::2]:
            if("-" in line):return False  
        return True
      ##################################
    def wrong_lines_add(lines):#function to find wrong lines in add problem
        lines=copy.deepcopy(lines)
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                if(lines[i][j]=="+"):lines[i][j]="0"
        for i in range(len(lines)):
            lines[i]=int("".join(lines[i]))
        sm=sum(lines[0:-1])
        Y1=list_lines[-1][0]########
        Y2=list_lines[-1][1]#########
        if(lines[-1]!=sm):return [(Y1,Y2,str(sm))]
        return []
      #####################################function to find wrong lines in sub problem
    def wrong_lines_sub(lines):
        lines=copy.deepcopy(lines)
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                if(lines[i][j]=="-"):lines[i][j]="0"
        for i in range(len(lines)):
            lines[i]=int("".join(lines[i]))
        dif=lines[0]-lines[1]
        Y1=list_lines[-1][0]
        Y2=list_lines[-1][1]
        if(lines[2]!=dif):return [(Y1,Y2,str(dif))]
        return []
      ########################################function to find wrong lines in mul problem
    def wrong_lines_mul(lines):
        lines=copy.deepcopy(lines)
        if("*" not in lines[0]):
            off,num1,num2,lines_off=(2,int("".join(lines[0])),int("".join(lines[1][1:])),lines[2:])
        else:
            idx=lines[0].index("*")
            off,num1,num2,lines_off=(1,int("".join(lines[0][0:idx])),int("".join(lines[0][idx+1:])),lines[1:])
        ans=[]
        for i in range(len(lines_off)):
            for j in range(len(lines_off[i])):
                if(lines_off[i][j]=="+" or lines_off[i][j]=="*"):lines_off[i][j]="0"
        for i in range(len(lines_off)):
            lines_off[i]=str(int("".join(lines_off[i])))
        exp_lines=find_lines_mul(num1,num2)
        for i in range(len(lines_off)):
            if(lines_off[i]!=exp_lines[i]):
                Y1=list_lines[i+off][0]
                Y2=list_lines[i+off][1]
                ans.append((Y1,Y2,exp_lines[i]))
        return ans
      ###########################################function to find wrong lines in div problem  
    def wrong_lines_div(lines):
        ans,lines=([],copy.deepcopy(lines))
        left_brk_idx=lines[0].index(")")
        right_brk_idx=lines[0].index("(")
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                if(lines[i][j]=="-"):lines[i][j]="0"

        num1,num2=( int("".join(lines[0][0:left_brk_idx])), int("".join(lines[0][left_brk_idx+1:right_brk_idx])) )
        for i in range(len(lines)):
            if(i==0):
                lines[0]=str(num1)+")"+str(num2)+"("+str(num2//num1)
            else:
                lines[i]=str(int("".join(lines[i])))
        exp_lines=find_lines_div(num2,num1)
        for i in range(len(exp_lines)):
            if(lines[i]!=exp_lines[i]):
                Y1=list_lines[i][0]
                Y2=list_lines[i][1]
                ans.append((Y1,Y2,exp_lines[i]))
        return ans
      ###########################################function to find expected lines in mul problem
    def find_lines_mul(num1,num2):
        ans=[]
        num2_list=num2list(num2)
        for n in range(len(num2_list)):
            ans.append(str(num1*num2_list[-1-n]*10**n))
        ans.append(str(num1*num2))
        return ans
      #############################################function to find expected lines in div problem
    def find_lines_div(num2,num1):
        zero_arr,quo,rem=([], num2//num1, num2%num1)
        if(num1>num2):return[str(num1)+")"+str(num2)+"("+str(quo),str(0),str(num2)]
        else:
            quo_list,num2_list,ans=(num2list(quo), num2list(num2), [])
            line1=str(num1)+")"+str(num2)+"("+str(quo)
            ans.append(line1)
            imm_num,idx=(0,0)
            while(idx<len(num2_list)):
                while(imm_num<num1 and idx<len(num2_list)):
                    imm_num=imm_num*10+num2_list[idx]
                    zero_arr.append(imm_num)
                    idx+=1
                if(imm_num>=num1):
                    ans.append(str(imm_num))
                    ans.append(str(num1*(imm_num//num1)))
                    imm_num=imm_num%num1
                else:idx+=1
            ans.append(str(imm_num))
            ans.pop(1)
            return ans

      ##########################################function to convert number to list of digits        
    def num2list(num):
        num_list=[]
        while(num!=0):
            num_list.append(num%10)
            num=(num-num%10)//10
        num_list=num_list[::-1]
        return num_list
      ############################################function to convert list of digits to number
    def list2num(lst):
        num=0
        for n in range(len(lst)):
            num+=lst[-1-n]*(10**n)
        return num
      #############################################
        
    if(only_one_op(lines,dic)):
        op=find_op(lines,dic)
        if(op=="add"):
            if(is_valid_add(lines,dic)):
                valid=True
                return (valid,wrong_lines_add(lines))
        elif(op=="sub"):
                if(is_valid_sub(lines,dic)):
                    valid=True
                    return (valid,wrong_lines_sub(lines))
        elif(op=="mul"):
                if(is_valid_mul(lines,dic)):
                    valid=True
                    return (valid,wrong_lines_mul(lines)) 
        elif(op=="div"):
                if(is_valid_div(lines,dic)):
                    valid=True
                    return (valid,wrong_lines_div(lines))

        return (valid,[])
    return (valid,[])   