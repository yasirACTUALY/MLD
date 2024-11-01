import numpy as np
import matplotlib.pyplot as plt


def Eval(X,W,Y):
    return ((np.sign(np.dot(X,W)) == Y).sum()/len(Y))

def PLA(X,Y,W):
    weights = W
    for i in range(0, len(X)):
        if np.dot(X[i], weights)*Y[i] <= 0:
            return weights+ Y[i]*X[i]
    return weights

def pocket(X,Y, W, n=1000):
    weights = W
    topWeights = W
    accuracy = Eval(X, weights, Y)
    for _ in range(0, n):
        weights = PLA(X,Y,weights)
        newAccuracy = Eval(X, weights, Y)
        if newAccuracy > accuracy:
            topWeights = weights
            accuracy = newAccuracy
    return topWeights

def getGradient(X, Y, w):
    # calculate the gradient
    gradient = np.zeros(X.shape[1])
    for i in range(0, X.shape[0]):
        div = np.clip((Y[i]*np.dot(w,X[i])), -250, 250)
        div = 1+np.exp(div)
        gradient += -1*Y[i]*X[i]/div
    return gradient/X.shape[0]

def LogisticRegressionAlgorithm(X,Y,lr=0.1):
    w = np.zeros(X.shape[1])
    for _ in range(0,1000):
        g = getGradient(X,Y,w)
        w = w - lr*g
        if np.linalg.norm(g) < 1e-6:
            break
    return w

def stochasticGradient(X,Y,W):
    # calculate the gradient
    div= (Y*np.dot(W,X))
    div = np.clip(div, -250, 250)  # Prevent exp overflow
    div = 1+np.exp(div)
    return -Y*X/div

def StochasticLogisticRegressionAlgorithm(X, Y, lr=0.1):
   w = np.random.rand(X.shape[1])
   
   # Initial learning rate
   initial_lr = lr
   
   for epoch in range(10000):
       # Decay learning rate over time
       current_lr = initial_lr / (1 + epoch / 1000)
       
       # select a random index
       i = np.random.randint(0, X.shape[0])
       
       # Update weights with decayed learning rate
       w = w - current_lr * stochasticGradient(X[i], Y[i], w)
   return w

def linearReg(X,Y):   
    XTranspose = np.transpose(X)
    XN = np.dot(XTranspose, X)
    XN = np.linalg.inv(XN)
    XL = np.dot(XN, XTranspose)
    return np.dot(XL, Y)

# function to find the vertial symmetry
def findSym(img):
    total = 0
    for i in range(0, 16):
        for j in range(0,16):
            # adds the difference between the pixel and x axis reflection
            total += abs(img[i*16+j] - img[((15-i)*16)+(j)])
        
    return total

def readFile(file, ones, fives):
    for line in file:
        if line[0] == '5':
            fives.append(line[7:-2].split(" "))
        elif line[0] == '1':
            ones.append(line[7:-2].split(" "))
    file.close()

def parse(AllFeatures, AllY, fives, ones):
    for i in range(0, len(fives)):
        img = fives[i]
        aveIntensity = np.average(img)
        sym = findSym(img)
        AllFeatures[i] = [1,aveIntensity, sym]
        AllY[i] = -1
    
    for i in range(0, len(ones)):
        img = ones[i]
        aveIntensity = np.average(img)
        sym = findSym(img)
        AllFeatures[i+len(fives)] = [1, aveIntensity, sym]
        AllY[i+len(fives)] = 1

def thirdOrder(XVals):
    transformed = []
    for _,x,y in XVals:
        transformed.append([1, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*(y**2)])
    return transformed

def plot_scatter(thirdOrderFeatures, fives):
    plt.scatter(thirdOrderFeatures[:len(fives),1], thirdOrderFeatures[:len(fives),2], c='r', label='fives', marker='x')
    plt.scatter(thirdOrderFeatures[len(fives):,1], thirdOrderFeatures[len(fives):,2], c='b', label='ones', marker='o', alpha=0.2)

def plot(W, Features, fives, label, title):
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.clf()
    plot_scatter(Features,fives)
    x = np.linspace(-1, .2, 100)
    yPok = -W[1]/W[2]*x - W[0]/W[2]
    plt.plot(x, yPok, label=label)
    plt.title(title)
    plt.legend()
    plt.show()


# loading training data
file = open('train.txt', 'r')
fives = []
ones = []
readFile(file, ones, fives)
fives = np.array(fives).astype('float')
ones = np.array(ones).astype('float')
twoFeatures = np.zeros((len(fives)+len(ones), 3), dtype='float')
AllY = np.zeros(len(fives)+len(ones))
parse(twoFeatures, AllY, fives, ones)
thirdOrderFeatures = thirdOrder(twoFeatures)
thirdOrderFeatures = np.array(thirdOrderFeatures).astype('float')
print(thirdOrderFeatures.shape)
print(twoFeatures.shape)

# loading test data
file=open('test.txt', 'r')
fivesTest = []
onesTest = []
readFile(file, onesTest, fivesTest)
print(len(fivesTest) + len(onesTest))
fivesTest = np.array(fivesTest).astype('float')
onesTest = np.array(onesTest).astype('float')
AllFeaturesTest = np.zeros((len(fivesTest)+len(onesTest), 3), dtype='float')
AllYTest = np.zeros(len(fivesTest)+len(onesTest))
parse(AllFeaturesTest, AllYTest, fivesTest, onesTest)
thirdOrderFeatureTest = thirdOrder(AllFeaturesTest)
thirdOrderFeatureTest = np.array(thirdOrderFeatureTest).astype('float')


# # plot the wlin
# wLin = linearReg(twoFeatures, AllY)
# wLinPok = pocket(twoFeatures, AllY, wLin)
# print("Ein for Pocket LR: {:.4f}% and Eout for Pocket LR: {:.4f}%".format((1-Eval(twoFeatures, wLinPok, AllY))*100, (1-Eval(AllFeaturesTest, wLinPok, AllYTest))*100))
# plot(wLinPok, thirdOrderFeatures, fives, 'wLinPocket', 'Logistic Regression With Pocket')

# # plot the wLRA
# w=LogisticRegressionAlgorithm(twoFeatures, AllY)
# print("Ein for LR with Gradient descent: {:.4f}% and Eout for LR with Gradient descent: {:.4f}%".format((1-Eval(twoFeatures, w, AllY))*100, (1-Eval(AllFeaturesTest, w, AllYTest))*100))
# plot(w, thirdOrderFeatures, fives, 'wLRA', 'Logistic Regression with Gradient Descent')

# wSt = StochasticLogisticRegressionAlgorithm(twoFeatures, AllY)
# print("Ein for Stochastic LR: {:.4f}% and Eout for Stochastic LR: {:.4f}%".format((1-Eval(twoFeatures, wSt, AllY))*100, (1-Eval(AllFeaturesTest, wSt, AllYTest))*100))
# plot(wSt, thirdOrderFeatures, fives, 'wSt', 'Stochastic Logistic Regression')





# Generate a meshgrid for x and y

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def plot_poly(X,labels, weights, initial, final, num):
    plt.clf()
    plt.scatter(X[:, 1][labels!=1], X[:, 2][labels!=1], color='blue', marker = '.')
    plt.scatter(X[:, 1][labels==1], X[:, 2][labels==1], color='red', marker = 'x')
    w = weights
    xx1 = np.linspace(initial[0], final[0], num)
    yy1 = np.linspace(initial[1], final[1], num)
    z = np.zeros((num, num))
    # AllFeatures[i] = [1,aveIntensity, sym,aveIntensity**2, sym**2, aveIntensity*sym,aveIntensity**3, sym**3,aveIntensity**2*sym,aveIntensity*sym*2]
    for i_x1, x in enumerate(xx1):
        for i_y1, y in enumerate(yy1):
            z[i_y1, i_x1] = np.sign(\
                w[0]*1 + \
                w[1]*(x) + w[2]*(y) + \
                w[3]*(x**2) + w[4]*(y**2) + w[5]*(y*x) + \
                w[6]*(x**3) + w[7]*(y**3) + w[8]*(y*(x**2)) +  w[9]*((y**2)*x))
    xx1, yy1 = np.meshgrid(xx1, yy1)
    plt.contour(xx1, yy1, z, levels=[0])
    plt.show()

X_raw = twoFeatures  # we only take the first two features.
Y = AllY
# Create poly featuresy
X = thirdOrderFeatures
# Plot
x_min, x_max = X_raw[:, 1].min() - .5, X_raw[:, 1].max() + .5
y_min, y_max = X_raw[:, 2].min() - .5, X_raw[:, 2].max() + .5


# plot the wlin
# wLin = linearReg(thirdOrderFeatures, AllY)
# wLinPok = pocket(thirdOrderFeatures, AllY, wLin)
# print("Ein for Pocket LR: {:.4f}%".format((1-Eval(X, wLinPok, Y))*100))
# print("Etest for Pocket LR: {:.4f}%".format((1-Eval(thirdOrderFeatureTest, wLinPok, AllYTest))*100))
# plot_poly(X_raw, Y, weights=wLinPok, initial=[x_min, y_min], final=[x_max, y_max], num=60)

w=LogisticRegressionAlgorithm(X, AllY,0.5)
print("Ein for LR with Gradient descent: {:.4f}%".format((1-Eval(X, w, Y))*100))
print("Etest for LR with Gradient descent: {:.4f}%".format((1-Eval(thirdOrderFeatureTest, w, AllYTest))*100))
plot_poly(X_raw, Y, weights=w, initial=[x_min, y_min], final=[x_max, y_max], num=60)

# wSt = StochasticLogisticRegressionAlgorithm(thirdOrderFeatures, AllY, 1)
# print("Ein for Stochastic LR: {:.4f}%".format((1-Eval(X, wSt, Y))*100))
# print("Etest for Stochastic LR: {:.4f}%".format((1-Eval(thirdOrderFeatureTest, wSt, AllYTest))*100))
# plot_poly(X_raw, Y, weights=wSt, initial=[x_min, y_min], final=[x_max, y_max], num=60)

plt.show()