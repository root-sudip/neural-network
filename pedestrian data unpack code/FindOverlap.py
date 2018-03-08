def get_region_code(pt,window,bool=True):
    #pt contains x,y
    #window contains xmin,ymin,xmax,ymax
    x=pt[0]
    y=pt[1]
    xmax=window[2]
    ymax=window[3]
    xmin=window[0]
    ymin=window[1]
    rcode=[False,False,False,False]
    #test Above
    if(y>=ymax):
        rcode[0]=True

    # test Below
    if (y < ymin):
        rcode[1] = True

    # test Right
    if (x >= xmax):
        rcode[2] = True

    #test Left
    if(x < xmin):
        rcode[3]=True

    if(bool):
        return rcode
    else:
        return [float(i) for i in rcode]

def test_line_clipping(line,window):
    #line is pt0, pt1 , pt as in get_region_code
    #window is similar
    pt0=line[0]
    pt1=line[1]
    rc0=get_region_code(pt0,window)
    rc1=get_region_code(pt1,window)
    #step 1
    #res = rc0 or rc1
    if(sum(rc0 or rc1)==0):
        #print("Whole line inside")
        return True
    #step 2
    elif(rc0 and rc1 != 0000):
        #print("Possible Partial Overlap")
        #print("whole line outside")
        return False
    else:
        return True

def test_two_rectangles(rect1,rect2):
    #rect is window
    #testing each line of rect1 with rect2 window
    xmin=rect1[0]
    ymin=rect1[1]
    xmax=rect1[2]
    ymax=rect1[3]
    we=[[xmin,ymax],[xmax,ymax]]
    ns=[[xmax,ymax],[xmax,ymin]]
    ew=[[xmax,ymin],[xmin,ymin]]
    sn=[[xmin,ymin],[xmin,ymax]]
    if(test_line_clipping(we,rect2)):
        return True
    elif(test_line_clipping(ns,rect2)):
        return True
    elif(test_line_clipping(ew,rect2)):
        return True
    elif(test_line_clipping(sn,rect2)):
        return True
    else:
        return False

def find_overlap(rect1,rect2):
    #rect2 is groundtruth
    #rect1 is sliding
    state = test_two_rectangles(rect1,rect2)
    if(not state):
        #print("Not Overlapping GT, But may cover GT...Testing Reverse")
        state = test_two_rectangles(rect2,rect1)
    if(state):
        #print("Overlap")
        return True
    else:
        #print("Non-Overlap")
        return False

#window = [128,384,192,448]
#view = [73,194,113,218]

#pt0=[150,400]
#pt1=[170,430]

#pt0=[200,430]
#pt1=[170,400]

#pt0=[150,350]
#pt1=[150,550]

#line=[pt0,pt1]
#rc= get_region_code(pt1,window)
#print(rc)

#test_line_clipping(line,window)
#rect1=[10,200,85,230]
#rect2=[73,194,113,218]
#find_overlap(rect1,rect2)