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

def delete_not_occuring_kmer(arr):
    for i in range(len(arr[0])):
        for j in arr:
            tmp=False
            if j[i]>0:
                tmp=True
                break
        if tmp==False:
            arr=np.delete(arr,i,1)
    return arr


def test():
    enhancers=read_file('enhancers_heart.fa')
    random=read_file('random.fa')
    kmers_in_seq=[]
    kmers=generate_kmers(4)
    y=[]
    for j in enhancers:
        k_n=[]
        for k in kmers:
            k_n.append(j.count(k))
        kmers_in_seq.append(k_n)
        y.append('enhancer')
    for j in random:
        k_n=[]
        for k in kmers:
            k_n.append(j.count(k))
        kmers_in_seq.append(k_n)
        y.append('random')
    kmers_in_seq=delete_not_occuring_kmer(np.array(kmers_in_seq))
    return kmers_in_seq, y


kmer,y=test()
