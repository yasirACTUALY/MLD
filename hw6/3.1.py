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
SEP = 10

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
        if sign < 0 and i > len(yValsBlue):
            w = w + X
            done = False
            i= 0
        elif sign > 0 and i < len(yValsBlue):
            w = w - X
            done = False
            i = 0

# use linear regression to find the weights
# create the matrix
# combine the x values  
X = np.zeros((2*N, 3))
for i in range(0, 2*N):
    if i < N:
        X[i][0] = 1
        X[i][1] = xValsRed[i]
        X[i][2] = yValsRed[i]
    else:
        X[i][0] = 1
        X[i][1] = xValsBlue[i-N]
        X[i][2] = yValsBlue[i-N]
Y = np.ones(2*N)
for i in range(0, N):
    Y[i] = -1
XTranspose = np.transpose(X)
XN = np.dot(XTranspose, X)
XN = np.linalg.inv(XN)
XL = np.dot(XN, XTranspose)
WLin = np.dot(XL, Y)

print(WLin.shape)
# graph the line with the weights w
x = np.linspace(-Rad-THK-5, (Rad+THK)*2, 100)
y = (-w[0] - w[1]*x)/w[2]
plt.plot(x, y, '-r', label='W-PLA')
print(w)

# graph the line with the weights wlin
y = (-WLin[0] - WLin[1]*x)/WLin[2]
plt.plot(x, y, '-b', label='W-Lin')




#  plot with colors
plt.scatter(xValsRed, yValsRed, c = 'b', label="Blue")
plt.scatter(xValsBlue, yValsBlue, c = 'r', label="Red")
plt.legend()
plt.show()
