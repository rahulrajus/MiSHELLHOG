import numpy as np
import cv2 as cv2
import math
from sklearn import svm

def get_hog_vector(test):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (4,4)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    # cv2.imshow('image',test)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    hist = hog.compute(test,winStride,padding,locations)
    return hist
positive_vector = []
negative_vector = []
labels = []#hi teacher
for i in range(1,16):
    print("/Users/Rahul/Google Drive/Personal_Projects/HOGTurtle/positive/track" +str(i).zfill(4)+".png")
    img = cv2.imread('./postive/track'+ str(i).zfill(4) + '.png')
    img = cv2.resize(img, (128,128))
#i failed my math final
    vctr = get_hog_vector(img)
    print(vctr)
    positive_vector.append(vctr)
    labels.append(1)
for i in range(1,68):
    print("/Users/Rahul/Google Drive/Personal_Projects/HOGTurtle/positive/track" +str(i).zfill(4)+".png")
    img = cv2.imread('./negative/negative'+ str(i).zfill(4) + '.png')
    img = cv2.resize(img, (128,128))
    vctr = get_hog_vector(img)
    negative_vector.append(vctr)
    labels.append(-1)
print(positive_vector)
print("negatives")
print(negative_vector)
train_vect = positive_vector + negative_vector
clf = svm.SVC()
train_vect = np.array(train_vect)
nsamples, nx, ny = train_vect.shape
train_vect = train_vect.reshape((nsamples,nx*ny))
clf.fit(np.array(train_vect), np.array(labels))
img_test = cv2.imread('./positive/track0001.png')
img = cv2.resize(img, (128,128))
#i failed my math final
vctr = get_hog_vector(img)
clf.predict(vctr)
