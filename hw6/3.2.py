import numpy as np
import matplotlib.pyplot as plt
# generate random values
import random
import math



# def red(x,rad,thk):
#     for i in range(0, len(x)):
#         l1 = math.sqrt(rad**2 - x[i]**2)
#         l2= math.sqrt((rad+thk)**2 - x[i]**2)
#         diff = abs(l1-l2)
#     l = np.square(x)
#     l= np.sqrt(rad-l)
#     val = random.random()
#     val *= thk
#     for i in range(0, len(l)):
#         if l[i] > 0:
#             l[i] = l[i] + val
#         else:
#             l[i] = l[i] - val
#     return l

# def Xvals(n, rad, thk):
#     xVals = np.random.rand(n)
#     xVals = xVals * 2* (rad + thk)
#     xVals = xVals - (rad+ thk)
#     return xVals




N = 1000
Rad = 10
THK = 5
SEPs = np.arange(0.2, 5.1, 0.2)
Itrs = []
for SEP in SEPs:
    xValsRed = []
    yValsRed = []

    for i in range(N):
        rad = random.uniform(Rad, Rad+THK)
        xval=random.uniform(-rad, rad)
        xValsRed.append(xval)
        yValsRed.append(math.sqrt(rad**2 - xval**2))

    xValsBlue = []
    yValsBlue = []


    for i in range(N):
        rad = random.uniform(Rad, Rad+THK)
        xval=random.uniform(-rad, rad)
        xValsBlue.append(xval + Rad + THK/2)
        yValsBlue.append(-math.sqrt(rad**2 - xval**2) - SEP)



    # run PLA on the dataset for red and blue
    iterations = 0
    w= np.ones(3)
    done = False
    while not done:
        done = True
        for i in range(0, len(yValsBlue) *2):
            x =0
            y= 0
            if i >= len(yValsBlue):
                x = xValsBlue[i - len(yValsBlue)]
                y = yValsBlue[i - len(yValsBlue)]
            else:
                x = xValsRed[i]
                y = yValsRed[i]
            X = np.array([1, x, y])
            sign = np.sign(np.dot(w,X))
            if sign < 0 and i >= len(yValsBlue):
                w = w + X
                iterations += 1
                done = False
                break
            elif sign > 0 and i < len(yValsBlue):
                w = w - X
                iterations += 1
                done = False
                break
    print("SEP = {}   Iterations = {}".format(SEP, iterations))
    Itrs.append(iterations)

    SEP += 0.2

plt.plot(SEPs, Itrs)
plt.xlabel('SEP')
plt.ylabel('Iterations')
plt.show()