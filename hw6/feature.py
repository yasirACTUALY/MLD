# load in the data from train.txt

import numpy as np
import matplotlib.pyplot as plt

# def findSym(img):
#     sym = 0
#     for i in range(0, 16):
#         sum = 0
#         for j in range(0,16):
#             total = 0
#             for z in range(j, 16):
#                 if img[i*16+z] > 0.2:
#                     total += img[i*16+z]
#                 else: break
#             if total > sum:
#                 sum = total
#         sym += sum
#     return sym


# def findSym(img):
#     total = 0
#     for i in range(0, 16):
#         mult = 1
#         for j in range(0, 16):
#             total += img[i+j*16] * mult
#             if img[i*16+j] > 0.5:
#                 mult =1
#             else:
#                 mult = 1
#     return total

# function to find the vertial symmetry
def findSym(img):
    total = 0
    for i in range(0, 16):
        for j in range(0,16):
            # adds the difference between the pixel and x axis reflection
            total += abs(img[i*16+j] - img[((15-i)*16)+(j)])
        
    return total

file = open('train.txt', 'r')
fives = []
ones = []
p = 0
for line in file:
    if line[0] == '5':
        fives.append(line[7:-2].split(" "))
    elif line[0] == '1':
        ones.append(line[7:-2].split(" "))

file.close()

fives = np.array(fives).astype('float')
ones = np.array(ones).astype('float')

# show an image of a 1
f0 = ones[0]
f0 = f0.reshape(16,16)
plt.imshow(f0, cmap='gray')
plt.show()

# show an image of a 5
f0 = fives[0]
f0 = f0.reshape(16,16)
plt.imshow(f0, cmap='gray')
plt.show()


fiveFeatures = np.zeros((len(fives), 2), dtype='float')
oneFeatures = np.zeros((len(ones), 2), dtype='float')
l = 0
ave = []
l1 = []
ave1 = []
for i in range(0, len(fives)):
    img = fives[i]
    aveIntensity = np.sum(img)
    sym = findSym(img)
    l1.append(sym)
    ave.append(aveIntensity)
    fiveFeatures[i] = [aveIntensity, sym]
for i in range(0, len(ones)):
    img = ones[i]
    aveIntensity = np.sum(img)
    sym = findSym(img)
    l += sym
    ave1.append(aveIntensity)
    # print("average intensity for one:{}".format(aveIntensity))
    oneFeatures[i] = [aveIntensity, sym]
# find the std dev of the symmetry
# l = np.std(l)
# l1 = np.std(l1)
# ave = np.std(ave)
# ave1 = np.std(ave1)
# print("Standard Deviation of symmetry for fives:{}  the mean for the symmety {}".format(l1, np.mean(l1)))
# print("Standard Deviation of symmetry for ones:{}  the mean for the symmety {}".format(l, np.mean(l)))
# print("Standard Deviation of average intensity for fives:{}  the mean for the average intensity {}".format(ave, np.mean(ave)))
# print("Standard Deviation of average intensity for ones:{}  the mean for the average intensity {}".format(ave1, np.mean(ave1)))
    
# print(fiveFeatures[:,0])
# print(fiveFeatures[:,1])
plt.scatter(fiveFeatures[:,0], fiveFeatures[:,1], c='r', label='fives', marker='x')
plt.scatter(oneFeatures[:,0], oneFeatures[:,1], c='b', label='ones', marker='o', alpha=0.2)

plt.xlabel('Intensity')
plt.ylabel('Symmetry')
plt.legend()

plt.show()






# run the code again to plot the test data
file = open('test.txt', 'r')
fives = []
ones = []
p = 0
for line in file:
    if line[0] == '5':
        fives.append(line[7:-2].split(" "))
    elif line[0] == '1':
        ones.append(line[7:-2].split(" "))

file.close()

fives = np.array(fives).astype('float')
ones = np.array(ones).astype('float')

# show an image of a 1
f0 = ones[0]
f0 = f0.reshape(16,16)
plt.imshow(f0, cmap='gray')
plt.show()

# show an image of a 5
f0 = fives[0]
f0 = f0.reshape(16,16)
plt.imshow(f0, cmap='gray')
plt.show()


fiveFeatures = np.zeros((len(fives), 2), dtype='float')
oneFeatures = np.zeros((len(ones), 2), dtype='float')
l = 0
ave = []
l1 = []
ave1 = []
for i in range(0, len(fives)):
    img = fives[i]
    aveIntensity = np.sum(img)
    sym = findSym(img)
    l1.append(sym)
    ave.append(aveIntensity)
    fiveFeatures[i] = [aveIntensity, sym]
for i in range(0, len(ones)):
    img = ones[i]
    aveIntensity = np.sum(img)
    sym = findSym(img)
    l += sym
    ave1.append(aveIntensity)
    # print("average intensity for one:{}".format(aveIntensity))
    oneFeatures[i] = [aveIntensity, sym]
plt.scatter(fiveFeatures[:,0], fiveFeatures[:,1], c='r', label='fives', marker='x')
plt.scatter(oneFeatures[:,0], oneFeatures[:,1], c='b', label='ones', marker='o', alpha=0.2)

plt.xlabel('Intensity')
plt.ylabel('Symmetry')
plt.legend()

plt.show()


