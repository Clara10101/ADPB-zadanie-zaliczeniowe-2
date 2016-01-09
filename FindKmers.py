def PSTable(W):
    PS=[-1]
    q=-1
    for x in W:
        while q>=0 and W[q]!=x:
            q=PS[q]
        q+=1
        PS.append(q)
    return PS

def KMP_MATCHER(T,P):
    n=len(T)
    m=len(P)
    PF=PSTable(P)
    q=0
    repeats=0
    for i in range(n):
        while q>0 and P[q]!=T[i]:
            q=PF[q]
        if P[q]==T[i]:
            q=q+1
        if q==m:
            repeats+=1
            q=PF[q]


    print(repeats,str([P,i+1-m]))
    return repeats

def Find(length_from,length_to,T):
    kmers=[]
    frame=len(T)-1
    repeats=[]
    d={}
    for i in range(len(T)-length_to):
        r=0
        lf=length_from


        while length_to>=lf:
            if T[i:i+lf] not in kmers:
                kmers.append(T[i:i+lf])
                wyn=KMP_MATCHER(T[i:i+frame],T[i:i+lf])
                r+=wyn
                lf+=1

                repeats.append(wyn)

            else: break
            if len(T[i:i+lf])-1 not in d:
                d[len(T[i:i+lf])-1]=wyn
            else:
                d[len(T[i:i+lf])-1]+=wyn
    return(d)

