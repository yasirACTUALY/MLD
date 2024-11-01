import numpy as np
from matplotlib import pyplot as plt

def sample(lb, ub, sz):
    return lb + np.random.random_sample((sz,))*(ub-lb)

def applyFunc(x, x1, x2):
    a = x1 + x2
    b = -x1 * x2
    return a*x + b

def outsampleErr(N,x1,x2):
    eout = []
    for _ in range(N):
        x = sample(-1, 1, 1)
        v = applyFunc(x, x1, x2)
        eout.append((v-x**2)**2)
    return np.mean(eout)

def var(N,x1list,x2list,varlist,biaslist):
    x1mean = np.mean(x1list)
    x2mean = np.mean(x2list)

    for i in range(len(x1list)):
        x1 = x1list[i]
        x2 = x2list[i]
        var = []
        bias = []
        for _ in range(N):
            x = sample(-1, 1, 1)
            g = applyFunc(x, x1, x2)
            gmean = applyFunc(x, x1mean, x2mean)
            var.append((g-gmean)**2)
            bias.append((gmean-x**2)**2)
        biaslist.append(np.mean(bias))
        varlist.append(np.mean(var))
    print("bias: ", np.mean(biaslist), "variance: ", np.mean(varlist), 'variance + bias: ', np.mean(varlist)+np.mean(biaslist))


def g(x,xavg,x2avg):
    return (xavg+x2avg)*x + (-xavg*x2avg)
def f(x):
   return x**2

vars = []
bias = []
err= []
x1vals = []
x2vals = []
for _ in range(1000):
    x1,x2 = sample(-1, 1, 2)
    x1vals.append(x1)
    x2vals.append(x2)
    err.append(outsampleErr(1000,x1,x2))

print('out of sample error: ', np.mean(err))
var(1000,x1vals,x2vals,vars, bias)

x = np.linspace(-1, 1, 100)
xavg = np.mean(x1vals)
x2avg = np.mean(x2vals)

plt.plot(x, f(x), color='red', label='f(x)=x^2')
plt.plot(x, g(x,xavg,x2avg), color='blue', label='gbar(x)')
plt.legend()

plt.show()