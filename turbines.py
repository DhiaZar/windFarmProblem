#import matplotlib.pyplot as plt
import random
import numpy as np
import math
import csv

turb = 46
gen = 2
n = turb + gen

space = 500
tol = 50
turbines = []


def dist(x,y,a,b):
    return math.sqrt((x-a)**2 + (y-b)**2)


while True:
    appending = True
    if len(turbines) == n:
        break
    (x,y) = (random.random()*space,random.random()*space)
    for j in turbines:
        if dist(x,y,j[0],j[1]) <= tol:
            appending = False
    if appending:
        turbines.append((x,y))

turbines_x = []
turbines_y = []
for i in range(n):
    turbines_x.append(turbines[i][0])
    turbines_y.append(turbines[i][1])

generator_x = [random.random()*space,random.random()*space]
generator_y = [random.random()*space,random.random()*space]

data = zip([i for i in range(n+gen)],turbines_x,turbines_y)
with open('Coordinates.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['i', 'x', 'y'])
    writer.writerows(data)

plt.scatter(turbines_x,turbines_y)
plt.scatter(turbines_x,generator_y,color='red')
plt.show()
        