import matplotlib.pyplot as plt
import random
import numpy as np
import math

turb = 46
gen = 2
n = turb

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

plt.scatter(turbines_x,turbines_y)
plt.scatter(generator_x,generator_y,color='red')
plt.show()
        