output=open("DaneTestowe.csv",'w')
import FindKmers as FK
def Extract():
    f=open("enhancers_heart.fa",'r')
    g=open("random.fa",'r')
    l=[]
    f=["attattattattcgggggggggggggattcggcgtatgcgatgccaattaaatttggg", 'qwertyuiopasdfghjklzxcvbnmqazwsxedcrfvtgbyhnujm']
    for line in f:
        #print(line)
        l.append(FK.Find(3,5,line.strip("\n")))

    for x in l:
        print(len(x))
    print(l)
Extract()