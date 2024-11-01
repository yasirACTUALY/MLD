import random
import matplotlib.pyplot as plt
import time
import numpy as np
# 1.10
# v1s = []
# vmins = []
# vrands = []
# start = time.time()
# v1Probs = np.zeros(100_000)
# vminsProbs = np.zeros(100_000)
# vrandsProbs = np.zeros(100_000)
# CoinsAmount = 1000
# FlipCount = 10
# flip_count_inv = 1 / FlipCount
# for _ in range(100_000):
#     min = -1
#     coins =[]
#     # Generate random bits in bulk to reduce overhead
#     random_bits = np.random.randint(0, 2, size=(CoinsAmount, FlipCount))
#     # print(f"random_bits: {random_bits.shape}")
    
#     # Calculate averages in bulk for all coins
#     coin_averages = random_bits.sum(axis=1) * flip_count_inv
#     # print(f"coin_averages: {coin_averages.shape}")
    
#     min_index = np.argmin(coin_averages)    
#     vmin = coin_averages[min_index]
#     # print(f"vmin: {vmin}")
#     v1 = coin_averages[0]
#     vrand = coin_averages[random.randint(0, CoinsAmount - 1)]
    
#     v1s.append(v1)
#     v1Probs[int(v1*10)] += 1
#     vmins.append(vmin)
#     vminsProbs[int(vmin*10)] += 1
#     vrands.append(vrand)
#     vrandsProbs[int(vrand*10)] += 1
# # label axis
# plt.hist([v1s,vmins,vrands],10, histtype='bar',   label= ['v1','vmin','vrand']) 
# plt.legend(loc='upper right')
# plt.xlabel('Probability')
# plt.ylabel('Count')
# plt.show()

# # PART C
# domain =  np.arange(0.0,0.6,0.1)
# v1P = 1
# vminP = 1
# vrandP = 1
# v1ps = np.zeros(6)
# vminps = np.zeros(6)
# vrandps = np.zeros(6)   
# v1s = np.array(v1s)
# vmins = np.array(vmins)
# vrands = np.array(vrands)
# v1s = np.abs(v1s-0.5)
# vrands = np.abs(vrands-0.5)
# vmins = np.abs(vmins-0.5)
# count = 0
# for i in range(len(v1s)):
#     if v1s[i] == 0:
#         count+= 1
# print(f"count: {count}")

# for i in range(domain.shape[0]):
#     epslon = domain[i]
#     v1ps[i] = np.sum(v1s > epslon)/100_000
#     vminps[i] = np.sum(vmins > epslon)/100_000
#     vrandps[i] = np.sum(vrands > i/10)/100_000
#     print(f"{i}v1p: {v1ps[i]}, vminp: {vminps[i]}, vrandp: {vrandps[i]}")   

# Huffman = 2*np.exp(-2*10*domain**2)
# plt.plot(domain,v1ps, label='v1')
# plt.plot(domain,vminps, label='vmin')
# plt.plot(domain,vrandps, label='vrand')
# plt.plot(domain,Huffman, label = 'Huffman')
# plt.ylabel('Probability')
# plt.xlabel('Epsilon')
# plt.legend(loc='upper right')
# plt.show()


# 1.7
import math
#     #n choose k
def nck(n,k):
    temp =math.factorial(k)*math.factorial(n-k)
    return math.factorial(n)/temp

def get_error(epsilon):
    error = 0
    for i in range (0,6):
        factor = i /6
        if(abs(factor - 0.5) > epsilon):
            print(f"i: {i}")
            p = nck(6,i)* 0.5**i *0.5**(6-i)
            error += p
    return error

domain =  np.arange(0.0,1.1,0.1)
vals = []
for i in range(domain.shape[0]):
    vals.append(get_error(domain[i]))
huffman = 2*np.exp(-2*6*domain**2)
plt.plot(domain,huffman,label='Hoeffding bound')
plt.plot(domain,vals, label='Error')
plt.ylabel('Error')
plt.xlabel('Epsilon')
plt.legend(loc='upper right')
plt.show()

# total = 0
# u = 0.9
# for k in range(13,26):
#     total += nck(25,k)*u**k * (1-u)**(25-k)
#     print(f"{k}")
# print(f"total: {total}")