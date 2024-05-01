import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import matplotlib
import gurobipy as grb
import time

cwd = os.getcwd()
dataPath = "BerkleyDataset"
path = os.path.join(cwd,dataPath)

try:
  os.mkdir(path, 0o666)
except:
  pass
import zipfile
with zipfile.ZipFile("Dataset.zip","r") as zip_ref:
    zip_ref.extractall(path)

path_img = os.path.join(cwd,"BerkleyDataset/Dataset/100075.jpg")
img = np.array(plt.imread(path_img), dtype="float64")
plt.imshow(img, cmap = 'gray')
plt.show()
(m,n) = np.shape(img)
N = 200
Z = np.random.normal(loc = 0, scale = np.sqrt(N), size = (m,n))
imgGaussian = img + Z
#normalizing in the interval float(0,1) and then multiplicate for 255 maybe it is a better choice
for i in range(m):
  for j in range(n):
    if(imgGaussian[i,j] < 0):
      imgGaussian[i,j] = 0
    elif(imgGaussian[i,j] > 255):
      imgGaussian[i,j] = 255
imgGaussian = imgGaussian.astype('uint8')
plt.imshow(imgGaussian, cmap = 'gray')
#plt.show()

y = np.array(imgGaussian)

def Frobenius_norm(X,Y):
    (m,n) = np.shape(X)
    (m2,n2) = np.shape(Y)
    if(m != m2):
        print("error")
    if(n != n2):
        print("error")
    norm = 0
    for i in range(m):
        for j in range(n):
            norm += X[i,j] * Y[i,j]
    norm = np.sqrt(norm)
    return norm


model = grb.Model('tv_reg')
model.params.NonConvex = 2
x = model.addMVar(
    (m,n),
    vtype = grb.GRB.CONTINUOUS,
    name = 'x'
)
z = model.addMVar(
    (m-1,n-1),
    vtype = grb.GRB.CONTINUOUS,
    lb = 0,
    name = 'z'
)
w = model.addMVar(
    (m,n),
    vtype = grb.GRB.CONTINUOUS,
    lb = 0,
    name = 'w'
)
rows = range(m)
columns = range(n)
dim = range(m*n)
""" # to be implemented
for i in range(m-1):
    for j in range(n-1):
        model.addConstr(z[i,j] == ((x[i+1,j] - x[i,j]) * (x[i+1,j] - x[i,j]) + (x[i,j+1] - x[i,j]) * (x[i,j+1] - x[i,j])))
"""

""" # to be implemented
for i in range(0,m-1):
    for j in range(0,n-1):
        model.addGenConstrPow(z[i,j],w[i,j], 0.5)
"""
valLambda = 0.4
expr = 0
expr += grb.quicksum((x[i,j] - y[i,j]) * (x[i,j] - y[i,j]) for i in rows for j in columns)
expr += valLambda * grb.quicksum(
     (x[i+1,j] - x[i,j]) * (x[i+1,j] - x[i,j]) + (x[i,j+1] - x[i,j]) * (x[i,j+1] - x[i,j]) for i in range(0,m-1) for j in range(0,n-1)
)

model.setObjective(expr,grb.GRB.MINIMIZE)
model.update()




#model.setParam('MIPgap', 0.01)
model.setParam(grb.GRB.Param.TimeLimit, 300)
model.setParam('OutputFlag', 1)

start = time.time()
model.optimize()
end = time.time()
comp_time = end - start
print(f"computational time: {comp_time} s")

for el in x:
    print(el)
img_recovered = np.zeros(shape = (m,n), dtype = 'uint8')

for i in range(m):
    for j in range(n):
        img_recovered[i,j] = int(x[i,j].X)
plt.imshow(img_recovered, cmap = 'gray')

print(f"difference between distorted and original: {Frobenius_norm(imgGaussian - img, imgGaussian - img)}")
print(f"difference between recovered and original: {Frobenius_norm(img_recovered - img, img_recovered - img)}")

plt.show()

plt.imshow(imgGaussian, cmap = 'gray')
plt.show()