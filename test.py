__author__ = 'MagdaTarka'

from itertools import product, chain
import pickle
import numpy as np

def read_file(a): #wczytanie danych z pliku
    p=open(a,'r')
    data=p.readlines()
    p.close()
    for i in range(len(data)):
        data[i]=data[i].strip()
    return data

def generate_kmers(k): #generuje wszystkie k-mery zlozone z liter A, C, T, G dlugosci k podanej jako parametr
    prod=product('ACTG',repeat=k)
    kmers=[]
    for i in prod:
        kmer=''
        for j in i:
            kmer=kmer+j
        kmers.append(kmer)
    return kmers

def test():
    enhancers=read_file('enhancers_heart.fa')
    random=read_file('random.fa')
    enh=open('enh.txt','w')
    ran=open('ran.txt','w')
    kmer_file=open('kmers.txt','w')
    km=[]
    e,r=[],[]
    for i in range(1,10):
        kmers=generate_kmers(i)
        km=km+kmers
        for k in kmers:
            number_e=0
            number_r=0
            for j in enhancers:
                number_e=number_e+j.count(k)
            for j in random:
                number_r=number_r+j.count(k)
            e.append(number_e)
            r.append(number_r)
    pickle.dump(np.array(e),enh)
    enh.close()
    pickle.dump(np.array(r),ran)
    ran.close()
    pickle.dump(np.array(km),kmer_file)
    kmer_file.close()

test()
