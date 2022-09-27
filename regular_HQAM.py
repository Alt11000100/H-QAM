from email.mime import base
from tkinter.tix import Y_REGION
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import random


def regular_hqam(m) :
    a, b = [], []
    if m%2==0 :
        c = 2**(m//2)
        x_coords = [[i for i in range(-c//2, c//2)] for j in range(c)]
        y_coords = [[0 for i in range(c)] for j in range(c)]
        for i in range(c) :
            for j in range(c) :
                if i%2==1 :
                    x_coords[i][j] += 0.5
                y_coords[i][j] += (c//2-i)*np.sqrt(3)/2 - np.sqrt(3)/4
        for i in range(c):
            for j in range(c):
                a.append(x_coords[i][j])
                b.append(y_coords[i][j])
        #print(a, b)
    else :
        if m==1 :
            a = [0, 1]
            b = [0, 0]
            x_coords = [[0, 1]]
            y_coords = [[0, 0]]
        elif m==3 :
            a = [-1, 0, 1, -0.5, 0.5, -1, 0, 1]
            b = [-np.sqrt(3)/2, -np.sqrt(3)/2, -np.sqrt(3)/2, 0, 0, np.sqrt(3)/2, np.sqrt(3)/2, np.sqrt(3)/2]
            x_coords = [[-1, 0, 1], [-0.5, 0.5], [-1, 0, 1]]
            y_coords = [[-np.sqrt(3), -np.sqrt(3), -np.sqrt(3)], [0, 0], [np.sqrt(3)/2, np.sqrt(3)/2], np.sqrt(3)/2]
        else :
            c = (2**((m-1)//2) + 2**((m+1)//2))//2
            x_coords = [[i for i in range(-c//2, c//2)] for j in range(c)]
            y_coords = [[0 for i in range(c)] for j in range(c)]
            for i in range(c) :
                for j in range(c) :
                    if i%2==1 :
                        x_coords[i][j] += 0.5
                    y_coords[i][j] += (c//2-i)*np.sqrt(3)/2 - np.sqrt(3)/4
            k = 2**(m//2 - 2)
            #print(k)
            l = len(x_coords)-1
            for i in range(k) :
                for j in range(k) :
                    x_coords[i].pop(0)
                    x_coords[i].pop(-1)
                    x_coords[l-i].pop(0)
                    x_coords[l-i].pop(-1)
                    y_coords[i].pop(0)
                    y_coords[i].pop(-1)
                    y_coords[l-i].pop(0)
                    y_coords[l-i].pop(-1)
            for i in range(c):
                for j, k in zip(x_coords[i], y_coords[i]):
                    a.append(j)
                    b.append(k)
    #print(a, b, x_coords, y_coords)
    return a, b, x_coords, y_coords
