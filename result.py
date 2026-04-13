
import ast


with open("FinalResult.txt","r") as f:
    a = f.read()
    sum= 0
    for i in a:
        sum += int(i)



print(sum)