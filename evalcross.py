import random
import subprocess as sp
for i in range(100):
 j=random.randrange(150,500)
 print(j)
 command="python rfcross.py 24_27rand.csv "+str(j)
 sp.call(command,shell=True)
 
