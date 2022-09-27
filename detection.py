from __future__ import division
from __future__ import print_function
import collections
import math
import numpy as np


from random import seed
from random import randint

from bisect import bisect_left


Point = collections.namedtuple("Point", ["x", "y"])




_Hex = collections.namedtuple("Hex", ["q", "r", "s"])
def Hex(q, r, s):
    assert not (round(q + r + s) != 0), "q + r + s must be 0"
    return _Hex(q, r, s)

def hex_add(a, b):
    return Hex(a.q + b.q, a.r + b.r, a.s + b.s)

def hex_subtract(a, b):
    return Hex(a.q - b.q, a.r - b.r, a.s - b.s)

def hex_scale(a, k):
    return Hex(a.q * k, a.r * k, a.s * k)

def hex_rotate_left(a):
    return Hex(-a.s, -a.q, -a.r)

def hex_rotate_right(a):
    return Hex(-a.r, -a.s, -a.q)

hex_directions = [Hex(1, 0, -1), Hex(1, -1, 0), Hex(0, -1, 1), Hex(-1, 0, 1), Hex(-1, 1, 0), Hex(0, 1, -1)]
def hex_direction(direction):
    return hex_directions[direction]

def hex_neighbor(hex, direction):
    return hex_add(hex, hex_direction(direction))

hex_diagonals = [Hex(2, -1, -1), Hex(1, -2, 1), Hex(-1, -1, 2), Hex(-2, 1, 1), Hex(-1, 2, -1), Hex(1, 1, -2)]
def hex_diagonal_neighbor(hex, direction):
    return hex_add(hex, hex_diagonals[direction])

def hex_length(hex):
    return (abs(hex.q) + abs(hex.r) + abs(hex.s)) // 2

def hex_distance(a, b):
    return hex_length(hex_subtract(a, b))

def hex_round(h):
    qi = int(round(h.q))
    ri = int(round(h.r))
    si = int(round(h.s))
    q_diff = abs(qi - h.q)
    r_diff = abs(ri - h.r)
    s_diff = abs(si - h.s)
    if q_diff > r_diff and q_diff > s_diff:
        qi = -ri - si
    else:
        if r_diff > s_diff:
            ri = -qi - si
        else:
            si = -qi - ri
    return Hex(qi, ri, si)

def hex_lerp(a, b, t):
    #not necessary
    return Hex(a.q * (1.0 - t) + b.q * t, a.r * (1.0 - t) + b.r * t, a.s * (1.0 - t) + b.s * t)

def hex_linedraw(a, b):
    #not necessary
    N = hex_distance(a,b)
    a_nudge = Hex(a.q + 1e-06, a.r + 1e-06, a.s - 2e-06)
    b_nudge = Hex(b.q + 1e-06, b.r + 1e-06, b.s - 2e-06)
    results = []
    step = 1.0 / max(N, 1)
    for i in range(0, N + 1):
        results.append(hex_round(hex_lerp(a_nudge, b_nudge, step * i)))
    return results

def complain(name):
    print("FAIL {0}".format(name))

def equal_int(name, a, b):
    if not (a == b):
        complain(name)
        
        
def equal_hex(name, a, b):
    if not (a.q == b.q and a.s == b.s and a.r == b.r):
        complain(name)        

def equal_hex_array(name, a, b):
    equal_int(name, len(a), len(b))
    for i in range(0, len(a)):
        equal_hex(name, a[i], b[i])

def test_hex_linedraw():
    equal_hex_array("hex_linedraw", [Hex(0, 0, 0), Hex(0, -1, 1), Hex(0, -2, 2), Hex(1, -3, 2), Hex(1, -4, 3), Hex(1, -5, 4)], hex_linedraw(Hex(0, 0, 0), Hex(1, -5, 4)))


#here are mine
def hex_centre(a):
    #to be implemented correctly
    #not used
     y_cartesian = (2 / 3) * np.sin(60) * (a.r - a.s)
     return y_cartesian

def pointy_hex_to_pixel(hex,offset_x=2,offset_y=0,size=2):
    #size is from centre to corner
   
    
    x = size*(math.sqrt(3) * hex.q  +  math.sqrt(3)/2 * hex.r)+offset_x
    y =  size*(                         3/2 * hex.r)+offset_y
    return (x,y)



def pointy_hex_to_pixel_regular(hex,offset_x=0,offset_y=0,size=math.sqrt(3)/3):
    #size is from centre to corner
    
    
    x = size*(math.sqrt(3) * hex.q  +  math.sqrt(3)/2 * hex.r)+offset_x
    y =  size*(                         3/2 * hex.r)+offset_y
    return (x,y)

def flat_pixel_to_hex(x,y,size):
    #this code works for flat hexagons need to be rotated
    cx = x/size
    cy = y/size 
    
    fx = (-2/3) * cx
    fy = (1/3)  * cx  +  (1/np.sqrt(3)) * cy
    fz = (1/3)  * cx  -  (1/np.sqrt(3)) * cy
    a = math.ceil(fx - fy)
    b = math.ceil(fy - fz)
    c = math.ceil(fz - fx)
    q = round((a - c) / 3)
    r = round((b - a) / 3)
    s = round((c - b) / 3)
    return [q,r,s]

def pointy_pixel_to_hex(x,y,size,hexd,x_offset=2,y_offset=0):
    #the actual detection 
    
    #size is from centre to corner
    
    x = x-x_offset
    y = y-y_offset
    
    cx = x/size
    cy = y/size 
    
    fx = (1/np.sqrt(3)) * cx -  (1/3)*cy
    fy = (2/3) * cy
    fz = -(1/np.sqrt(3))* cx  -  (1/3) * cy
    a = math.ceil(fx - fy)
    b = math.ceil(fy - fz)
    c = math.ceil(fz - fx)
    q = round((a - c) / 3)
    r = round((b - a) / 3)
    #s = round((c - b) / 3)
    s = -q-r
    #return [-s,-r,-q]
    #return Hex(-s,-r,-q)
    
    hextmp = Hex(q,r,s)
    
    if(hexd.get(hextmp) != None):
        return hextmp
    else:
        #check for the closer neighboor
        #won't work for low SNR
        #t = list(hexd.items())
        
        ret = [10000]*6
        for i in range(6):
            s = hexd.get(hex_neighbor(hextmp, i))
            
            if(s != None):
                
                ret[i] = (np.sqrt(  (x-s[0])**2 + (y-s[1])**2) )
        
        min_ind = [ x for x in range(len(ret)) if ret[x] == min(ret) ]
        
        if(len(min_ind)!=0 and min(ret)!=10000):
             return hex_neighbor(hextmp, min_ind[0])
        
         
    
    #that's a prob
    return Hex(0,0,0)
    
    
    #return Hex(q,r,s)

def pointy_pixel_to_hex_regular(x,y,size,x_offset=0,y_offset=0):
    #size is from centre to corner
    
    x = x-x_offset
    y = y-y_offset
    
    cx = x/size
    cy = y/size 
    
    fx = (1/np.sqrt(3)) * cx -  (1/3)*cy
    fy = (2/3) * cy
    fz = -(1/np.sqrt(3))* cx  -  (1/3) * cy
    a = math.ceil(fx - fy)
    b = math.ceil(fy - fz)
    c = math.ceil(fz - fx)
    q = round((a - c) / 3)
    r = round((b - a) / 3)
    s = round((c - b) / 3)
    #s = -q-r
    #return [-s,-r,-q]
    return Hex(q,r,s)
    
def detect_regular(r,s,x_offset=0,y_offset=0,size=math.sqrt(3)/3):
    d = [0]*len(s)
    for i in range(len(s)):
        d[i]= pointy_pixel_to_hex_regular(r[i], s[i], size,x_offset,y_offset)
    return d   
    
   
def hex_ring(center, radius,m):
    results = []
    # this code doesn't work for radius == 0; 
    hex = hex_add(center,hex_scale(hex_direction(4), radius))
    for i in range(6):
        for j in range(radius):
            results.append(hex)
            hex = hex_neighbor(hex, i)
    return results 
   
def hex_spiral(center, radius,m):
    results = []
    #radius = math.floor(m/6)
    
    for k in range(1,radius):
        results.append(hex_ring(center, k,6))
        
    #results.append(hex_ring(center,radius+1,m%6)) 
    
    return results



def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1

def how_many(n):
    n=n-1
    d = (1) + (4*n/3)
    
    sol1 = (-1-np.sqrt(d))/(2)  
    sol2 = (-1+np.sqrt(d))/(2)
    
    if sol1>0:
        return sol1
    else :
        return sol2
 
    
def add_awgn_axis(array,sigma=0.6,mu=0):
    q = []
    n_samples = len(array)

    s = np.random.normal(mu,sigma, n_samples)
    for i in range(n_samples):
        q.append(array[i]+s[i])
    
    return  q

def detect(r,s,hexd,x_offset,y_offset,size=2):
    #main detection scheme
    d = [0]*len(s)
    for i in range(len(s)):
        d[i]= pointy_pixel_to_hex(r[i], s[i], size,hexd,x_offset,y_offset)
    return d  
    

def avg_pwr(r,s):
    p = [0]*len(r)
    for i in range(len(r)):
        p[i] = (r[i]**2+s[i]**2)
    return np.mean(p)

def convert_dB(x):
    return 10.0 * np.log10(x)

def rnd_samples(r,s,samples=100):
    q = []
    t = []
    seed(1)
    for _ in range(samples):
        tmp = randint(0,len(r)-1)
        q.append(r[tmp])
        t.append(s[tmp])
    return q, t
    

def SNR_dB(r,s,value,samples=100):
    #return with awgn and without
     tmp1 = []
     tmp2 = []
     tmp1 ,tmp2 = rnd_samples(r,s,samples)
     
     temp = convert_dB(avg_pwr(r,s)) - value
     
     
     x = 10.0 ** (temp / 10.0) /2
     
     
     
     return add_awgn_axis(tmp1,x**(1/2)) , add_awgn_axis(tmp2,x**(1/2)) , tmp1 , tmp2
     
def SEP_pixel(a,b,c,d):
    #check how many are the same
    #percentage / how many mistakes
    mistakes = 0
    for i in range(len(a)):
        
        if(a[i] == c[i] and b[i] == d[i]):
            continue
        else :
            mistakes = mistakes + 1
    return mistakes / len(a) , mistakes

def SEP_hex():
    pass


def offset(M,size,w,h):
    #returns x_offset,y_offset
    if(M==4):
        return w/2 , 0
    elif(M==8):
        return 0 , h/16
    elif(M==16):
        return w/4 , 0
    elif(M==32):
        return w/8,0
    elif(M==64):
        return w/2 , 0
    elif(M==128):
        #could be wrong
        return w/2 , 0
    elif(M==256):
        return w/8 , 0
    elif(M==512):
        return w/8 , 0
    elif(M==1024):
        return w/8 , 0
    else :
        return w/2 , 0
        
        
        
        
            
        
    
#a = Hex(0.0, 0.0, 0.0)
#b = Hex(1.0, -1.0, 0.0)
#c = Hex(0.0, -1.0, 1.0)

#print(hex_neighbor(b, 2))
#print(hex_distance(a, c))


