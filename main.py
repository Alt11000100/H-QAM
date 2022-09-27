import matplotlib.pyplot as plt
import detection as st
import math

import numpy as np
from time import process_time
from timeit import default_timer as timer

import matplotlib.patches as ptc

import regular_HQAM as re

from scipy import special




a = st.Hex(0.0, 0.0, 0.0)











#some helping methods:
def distfromcentre(a,x_offset,y_offset):
    #given a hex calculates the distance from 0,0
    
    
    x = st.pointy_hex_to_pixel(a,x_offset,0)[0]
    y = st.pointy_hex_to_pixel(a,0,y_offset)[1]
    
    return (x**2+y**2)**(1/2)






def get_distance(p, q):
    """ 
    Return euclidean distance between points p and q
    assuming both to have the same number of dimensions
    """
    # sum of squared difference between coordinates
    s_sq_difference = 0
    for p_i,q_i in zip(p,q):
        s_sq_difference += (p_i - q_i)**2
    
    # take sq root of sum of squared difference
    distance = s_sq_difference**0.5
    return distance


def my_mld_simple(hexd,symbol):
    #sth = []
    sth = list(hexd.values())
    a = []
    k = len(sth)
    
    
    for i in range(len(sth)):
        a.append(get_distance(symbol, sth[i]))
    
    min_val_idxs = [ x for x in range(k) if a[x] == min(a)]
    
    return sth[min_val_idxs[0]]

def my_mld_simple_SEP(r,s,f1,f2,hexd):
    
    d1=[0]*len(r)
    d2 =[0]*len(r)
    
          
    for i in range(len(r)):
        
        d1[i] ,d2[i] = my_mld_simple(hexd,[r[i],s[i]])
    
    
    
    return st.SEP_pixel(d1, d2, f1, f2)

def distance1(r,s,c):
    #not to be used
    k = len(r)
    #k=10
    a = [0]*k
   
    for i in range(k):
        a[i] = get_distance([r[i],s[i]],c)
    
    
    min_val_idxs = [ x for x in range(k) if a[x] == min(a)]
    
   # print(r[min_val_idxs[0]])
    #print(s[min_val_idxs[0]])
    #print(st.pointy_pixel_to_hex(r[min_val_idxs[0]], s[min_val_idxs[0]], 2))
    
    return min_val_idxs[0]
    
    
        
    

def simple_mld(r,s,r1,s1):
    #not to be used
    k = len(r1)
    #k = 10
    a =[0]*k
    
    for i in range(k):
       a[i] =  distance1(r,s,[r1[i],s1[i]])
    
    
    
    return a




#================================================================


def plotplot(r,q,d,m=64):
    
    
    fig, ax2 = plt.subplots()

    ax2.spines['left'].set_position('zero')
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks([])

    ax2.scatter(r, q, s=2000 / m, c='#2494DF', linewidths=0.6, edgecolors='black')

    for i in range(len(r)): #9*M/8
        hex2 = ptc.RegularPolygon([r[i], q[i]], numVertices=6, radius=np.sqrt(3) * d / 3,
                                       orientation=np.radians(0),
                                       facecolor='b', fill=False, alpha=0.4, edgecolor='#0C4097')
        ax2.add_patch(hex2)
    
    plt.show()


#plotplot(r, s, w)

#print(get_distance(hexd[st.Hex(0,0,0)], hexd[st.Hex(0,-1,1)]))


#plt.grid(True)
#plt.scatter(r,s)
#plt.scatter(r1,s1, s=2000 / m, c='#2494DF', linewidths=0.6, edgecolors='black')
#plt.scatter(r2,s2)


#plt.scatter(s,r)

#plt.plot(r,'-o',color='blue',label='No mask')
#plt.plot(s,'-x',color='red',label='No mask')


#=============================REGULAR

def Reg_HQAM(exp):
    #works for exp>3
    a , b, c , d= re.regular_hqam(exp)
    
    if(exp!=5):
        y_off = math.sqrt(3)/4
    elif(exp==5):
        y_off = -math.sqrt(3)/4
    else:
        y_off = 0
    
    d2 = st.detect_regular(a,b,0,-y_off,math.sqrt(3)/3)
    
    
    
    
    #plt.scatter(a,b)
    
    r2 = []
    s2 = []
    
    hexd = {}
    
    for i in range(len(d2)):
        
     
        r2.append(st.pointy_hex_to_pixel_regular(d2[i],0,-y_off)[0])
        
        #r2.append(hexd.get(d2[i])[0])
               
        s2.append(st.pointy_hex_to_pixel_regular(d2[i],0,-y_off)[1])
        #s2.append(hexd.get(d1[i])[1])
        
        hexd[d2[i]] = r2[-1] , s2[-1]
    
    
    #plt.scatter(r2,s2)
    plotplot(r2,s2,1,2**exp)
    
    return r2,s2,hexd,0,y_off
    
#============================

#Reg_HQAM(6)
#===============================

def Irr_HQAM(M):
    #the basic constallation
    size=2
    w=size*((3)**(1/2))
    h=2*size
    
    x_offset , y_offset = st.offset(M, size, w, h)
    
    a = st.Hex(0.0, 0.0, 0.0)

    #we create r,s which will be the x,y coordinates to plot
    s = []
    r= []
    hextry = []
    #these are all
    number = M
    
    
    n = math.ceil(st.how_many(number))
    
    
    #these are the actual symbols
    hextry = st.hex_spiral(a,n+5,17)
    
    
    hextry_flat = hextry.copy()
    
    
    #print(hextry)
    
    #let's define a dictionary for easy retrieval
    
    hexd = {}
    
    
    #===================================
    hextry_flat1 = []
    
    for j in range(len(hextry_flat)):
        for i in range(len(hextry_flat[j])):
            hextry_flat1.append(hextry_flat[j][i])
    
    hextry_flat1.append(st.Hex(0,0,0))
    
    
    #bubblesort to discard high energy symbols
    
    for i in range(len(hextry_flat1)):
        for j in range(0,len(hextry_flat1)-i-1):
            if(distfromcentre(hextry_flat1[j],x_offset,y_offset) > distfromcentre(hextry_flat1[j+1],x_offset,y_offset)):
                       hextry_flat1[j] , hextry_flat1[j+1] = hextry_flat1[j+1], hextry_flat1[j]
                   
                   
    for i in range(len(hextry_flat1)-number):
        hextry_flat1.pop()
    
    
    
    
    for i in range(len(hextry_flat1)):
        r.append(st.pointy_hex_to_pixel(hextry_flat1[i],x_offset,y_offset)[0])
                    
        s.append(st.pointy_hex_to_pixel(hextry_flat1[i],x_offset,y_offset)[1])
            
        hexd[hextry_flat1[i]] = r[-1] , s[-1]  

    #plotplot(r, s, w,M)
    
    return r,s,hexd,x_offset,y_offset

#Irr_HQAM(256)

def my_detection(r1,s1,r10,s10,hexd,samples,SNR,x_offset,y_offset,only_SEP=True,Regular=False):
    
    r2 = []
    s2 = []
    
    #r1 , s1 ,r10 , s10 = st.SNR_dB(r, s, SNR ,samples)
    
    start = timer()
    if(Regular == False):
        d1 = st.detect(r1,s1,hexd,x_offset,y_offset)
    else:
        d1 = st.detect(r1, s1,hexd,x_offset,-y_offset,math.sqrt(3)/3)
        
    end = timer()
    
    print("This is detection time:",end-start)
    
    
    for i in range(len(d1)):
        
     
        #r2.append(st.pointy_hex_to_pixel(d1[i],x_offset,y_offset)[0])
        
        r2.append(hexd.get(d1[i])[0])
               
        #s2.append( st.pointy_hex_to_pixel(d1[i],x_offset,y_offset)[1])
        s2.append(hexd.get(d1[i])[1])
    
    my_SEP = st.SEP_pixel(r10,s10,r2,s2)[0]
    
    if(only_SEP):
        
        return my_SEP
    
    return d1 , r2 , s2 , my_SEP


def Thrasso_SEP(M, d, SNR):            #for regular hqam
    
    ext_symbs = 0             #external symbols
    so=int(math.log2(M))
    
    constellation0,constellation1 ,s,t,ole= Reg_HQAM(so)
    
    
    
    Es_total = 0

    for i in range(len(constellation0)):
        Es_total += constellation0[i] ** 2 + constellation1[i] ** 2

    Es = Es_total / M

    N0 = Es / (np.power(10, SNR / 10))
    #print("N0 = ", N0)

    R_in = d/2                    #hexagonal's incircle radius
    R_ext = d/np.sqrt(3)          #hexagonal's circumcircle radius

    k = 0
    opt_k = 0                 #optimum k
    if M == 16:
        ext_symbs = 10
        opt_k = 0.8711505
    if M == 32:
        ext_symbs = 13
        opt_k = 0.7233274
    if M == 64:
        ext_symbs = 22
        opt_k = 0.5222431
    if M == 128:
        ext_symbs = 27
        opt_k = 0.5088351
    if M == 256:
        ext_symbs = 46
        opt_k = 0.3936315
    if M == 512:
        ext_symbs = 64
        opt_k = 0.3672311
    if M == 1024:
        ext_symbs = 128
        opt_k = 0.2982858

    #k = opt_k
    k=0
    r = k*R_ext + (1-k)*R_in          #ρ = kR' + (1-k)R

    thrasso_SEP = ((2*M-ext_symbs)/(2*M))*np.power(np.e, -(r**2)/N0) + ext_symbs/M*(1/2*special.erfc(r/np.sqrt(N0)))         #Q(x) = 1/2erfc(x/sqrt(2))

    # min = opt.minimize(TSEP, 0.75, bounds=((0, 1),))
    # return min.x

    return thrasso_SEP


def diagram1(M=128,reg=True,char='-o',color='blue',label='Approx [1]',label1=None):
    
    
    #label1 = 'HQAM Approx [1]'
    
    
    y_sep = []
    
    y_sep_thrass =[]
    
    if(reg):
        so=int(math.log2(M))
        
        r , s , hexd ,x_offset,y_offset = Reg_HQAM(so)
    else:
        
        r , s , hexd ,x_offset,y_offset = Irr_HQAM(M)
    
    
    
    samples = 100000
    helpi = 1/(samples)
   
    
    for i in range(0,45):
        r1 , s1 ,r10 , s10 = st.SNR_dB(r, s, i ,samples)
        
       # y_sep.append(my_mld_simple_SEP(r1,s1,r10,s10,hexd)[0])
        #y_sep_thrass.append(Thrasso_SEP(M, 1, i))
         
        y_sep.append(my_detection(r1, s1, r10, s10, hexd, samples, i, x_offset, y_offset,True,reg))
        if(y_sep[-1]<helpi #or y_sep_thrass[-1]<helpi
           ):
            break
    
    
    plt.plot(y_sep,char,color=color,label=label)
    #plt.plot(y_sep_thrass,'--',color='black',label=label1)
    
    pass

def diagram2(M=128,char='-o',color='blue',label='proposed'):
    
    
    
    
    y_sep = []
    
    r , s , hexd ,x_offset,y_offset = Irr_HQAM(M)
   
    samples = 100000
    
   
    
    for i in range(0,40):
        r1 , s1 ,r10 , s10 = st.SNR_dB(r, s, i ,samples)
        
        y_sep.append(my_mld_simple_SEP(r1,s1,r10,s10,hexd)[0])
         
        #y_sep.append(my_detection(r1, s1, r10, s10, hexd, samples, i, x_offset, y_offset))
    
    
    
    plt.plot(y_sep,char,color=color,label=label)
    
    pass


def diagram():
    #make the SNR/SEP diagram for irregular or regular(reg) hqam
    fig = plt.subplots()
    plt.gca().set_yscale('log')
    
    #plt.axis([10, 40, 0.00001, 1])
    diagram1(16,False,'--','black','16-HQAM(Sim)')
    diagram1(32,False,'--','black','32-HQAM(Sim)')
    #diagram1(64,True,'--','red','64-HQAM(Sim)_reg')
    #diagram1(64,False,'--','black','64-HQAM(Sim)')
    #diagram1(128,True,'--','red','128-HQAM(Sim)_reg')
    #diagram1(256,True,'--','red','256-HQAM(Sim)_reg')
    #diagram1(512,True,'--','red','512-HQAM(Sim)_reg')
    #diagram1(128,False,'--','black','128-HQAM(Sim)')
    #diagram1(256,False,'--','black','256-HQAM(Sim)')
    #diagram1(512,False,'--','black','512-HQAM(Sim)')
    #diagram1(1024,False,'--','black','1024-HQAM(Sim)')
    #diagram1(1024,True,'--','red','1024-HQAM(Sim)_reg')
    #diagram1(2048,False,'--','black','2048-HQAM(Sim)')
    
    #diagram1(128,True,'^','red','128-HQAM (Sim)')
    #diagram2(16,'-x','#33dda5','MLD 16-HQAM (Sim)')
    #diagram1(32,'o','black','Proposed 32-HQAM (Sim)')
    #diagram2(32,'-x','#aaccbb','Proposed Detection 32-HQAM Simulated')
    #diagram1(64,'x','black','Proposed 64-HQAM (Sim)')
    #diagram1(128,'*','black','Proposed 128-HQAM (Sim)')
    #diagram1(256,'-*','black','Proposed 128-HQAM (Sim)')
    #diagram1(512,'-*','black','Proposed 128-HQAM (Sim)')
    #diagram1(1024,'-p','black','Proposed 16-HQAM (Sim)')
      
    plt.legend(loc=3,fontsize=7)
    
    plt.xlabel(xlabel='SNR (dB)')
    plt.ylabel(ylabel='SEP')
    pass

#Reg_HQAM(5)

#print(Thrasso_SEP(64, 1/2, 20))

diagram()


    
