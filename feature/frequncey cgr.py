import collections
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib import cm
import pylab
import math
import os
import sys
import cv2
import numpy as np
#from Bio import SeqIO
import re


def count_kmers(sequence, k):
    d = collections.defaultdict(int)
    for i in range(len(sequence)-(k-1)):
        d[sequence[i:i+k]] +=1
    for key in d.keys():
        if "N" in key:
            del d[key]
    return d

 

def probabilities(sequence, kmer_count, k):
    probabilities = collections.defaultdict(float)
    N = len(sequence)
    for key, value in kmer_count.items():
        probabilities[key] = float(value) / (N - k + 1)
    return probabilities

 

def chaos_game_representation(probabilities, k):
    array_size = int(math.sqrt(4**k))
    chaos = []
    for i in range(array_size):
        chaos.append([0]*array_size)

    maxx = array_size
    maxy = array_size
    posx = 1
    posy = 1
    for key, value in probabilities.items():
        for char in key:
            if char == "T":
                posx += maxx / 2                
            elif char == "C":
                posy += maxy / 2
            elif char == "G":
                posx += maxx / 2
                posy += maxy / 2
            maxx = maxx / 2
            maxy /= 2

        chaos[int(posy-1)][int(posx-1)] = value
        maxx = array_size
        maxy = array_size
        posx = 1
        posy = 1

    return chaos



def save_fcgr(id, sequence, k):
    chaos = chaos_game_representation(probabilities(str(sequence), count_kmers(str(sequence), k), k), k)    

    # show with 
    pylab.figure(figsize=(5,5))
    pylab.imshow(chaos, cmap=cm.gray_r) 
    pylab.axis('off')
    pylab.xticks([])
    pylab.yticks([])
    #pylab.show()
    pylab.savefig(str(k) + '.png', dpi = 100,bbox_inches='tight',pad_inches = 0)
    pylab.close()

def list_fcgr(id, sequence, k):
    chaos = chaos_game_representation(probabilities(str(sequence), count_kmers(str(sequence), k), k), k)    

    chaos = np.array(chaos)
    fcgr = chaos.flatten()
    np.savetxt('work/fcgrlist/%s.txt'%id, fcgr,fmt='%f',delimiter=',')  