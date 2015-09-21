from numpy import *

from matplotlib.pyplot import *



colors = 'rbcm'
markers = 'soos'
def show_pts(data):
    for i in range(4):
        idx = (arange(npts) % 4 == i)
        plot(data[0,idx], data[1,idx],
             marker=markers[i], linestyle='.',
             color=colors[i], alpha=0.5)
    gca().set_aspect('equal')

def show_pts_1d(data):
    for i in range(4):
        idx = (arange(npts) % 4 == i)
        plot(data[idx], marker=markers[i], linestyle='.',
             color=colors[i], alpha=0.5)
    gca().set_aspect(npts/4.0)

#### Copy in your implementation from Assignment #1 ####
def sigmoid(x):
    return 0 # dummy
#### or if the starter code is posted, uncomment the line below ####
# from nn.math import sigmoid

npts = 4 * 40; random.seed(10)
x = random.randn(npts)*0.1 + array([i & 1 for i in range(npts)])
y = random.randn(npts)*0.1 + array([(i & 2) >> 1 for i in range(npts)])
data = vstack([x,y])
figure(figsize=(4,4)); show_pts(data); ylim(-0.5, 1.5); xlim(-0.5, 1.5)
xlabel("x"); ylabel("y"); title("Input Data")


