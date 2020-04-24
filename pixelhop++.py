from saab import Saab
import numpy as np
from glob import glob
import cv2
import os
from skimage.util import view_as_windows
from skimage.measure import block_reduce
import csv
import sklearn
from sklearn.svm import SVC

def MaxPooling(x):
    return block_reduce(x, (1, 2, 2, 1), np.max)

def Shrink(X, win,stride):
    X = view_as_windows(X, (1,win,win,1), (1,stride,stride,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def PixelHop_Unit(X,num_kernels,saab=None,window=5,stride=1,train=True):
    print('input shape',X.shape )
    X = Shrink(X, 5,1)
    print('extracting patches',X.shape)
    S = list(X.shape)
    X = X.reshape(-1, S[-1])
    if(train==True):
        saab = Saab(num_kernels=num_kernels, useDC=True, needBias=True)
    saab.fit(X)
    transformed = saab.transform(X).reshape(S[0],S[1],S[2],-1)
    print('transformed shape',transformed.shape)
    return saab,transformed

def faltten(listoflists,kernel_filter):
    flattened=[]
    for i in range(len(listoflists)):
        for j in range(len(listoflists[i][kernel_filter[i]])):
            flattened.append(listoflists[i][j])
    return flattened

def PixelHopPP_Unit(X,num_kernels,saab=None,window=5,stride=1,train=True,energy_th=0,ch_decoupling=True,ch_energies=None,kernel_filter=[]):
    #for taining specify X, num_kernels, window, stride, train, energy_th, ch_decoupling, ch_energies(only if it is not the first layer)
    #for testing specift X, num_kernels, saab, window, stride, train, ch_decoupling, kernel_filter
    N,L,W,D=X.shape
    if(ch_energies==None):
        ch_energies=np.ones((D)).tolist()
    out_ch_energies=[]
    output=None
    if(ch_decoupling==True):
        for i in range(D):
            saab,transformed=(PixelHop_Unit(X[:,:,:,i].reshape(N,L,W,1),num_kernels=num_kernels,saab=saab, window=window,stride=stride,train=train))
            if(train==True):
                out_ch_energies.append(ch_energies[i]*saab.Energy)
                kernel_filter.append(out_ch_energies[i]>energy_th)
            transformed=transformed[:,:,:,kernel_filter[i]]
            print(i,'transformed shape',transformed.shape)
            if(i==0):
                output=transformed
            else:
                output=np.concatenate((output,transformed),axis=3)

    else:
        saab,transformed=(PixelHop_Unit(X,num_kernels=num_kernels,saab=saab ,window=window,stride=stride,train=train))
        if(train==True):
            out_ch_energies.append(saab.Energy)
            kernel_filter.append(out_ch_energies[0]>energy_th)
        transformed=transformed[:,:,:,kernel_filter[0]]
        output=transformed
    print('final output shape',output.shape)
    return saab,output,kernel_filter,faltten(out_ch_energies,kernel_filter)#ch_energies is not needed

path= "/home/mozhdeh/lfwppp/HEFrontalizedLfw2/"
pics = glob(path+"*")
size=32
num=13111
raw_images=np.zeros((num,size,size,1))
flipped_raw_images=np.zeros((num,size,size,1))
raw_labels=[]
for i in range (num):
    image = cv2.imread(pics[i])
    image = cv2.resize(image,(32,32))
    raw_images[i,:,:,0]=image[:,:,0]/255
    flipped_raw_images[i,:,:,0]=cv2.flip( image, 1 )[:,:,0]/255
    raw_labels.append(os.path.basename(pics[i]).split('\\')[-1].split('.')[0])

raw_labels=np.asarray(raw_labels)
print(image.shape)


numoftrainpixelhop=4000
energy_th = .0005
saab,transformed,kernel_filter,falttened=PixelHopPP_Unit(raw_images[0:numoftrainpixelhop],num_kernels=18,saab=None,window=5,stride=1,train=True,energy_th=energy_th,ch_decoupling=False,ch_energies=None,kernel_filter=[])
_,out1,_,_=PixelHopPP_Unit(raw_images,num_kernels=18,saab=saab,window=5,stride=1,train=False,ch_decoupling=False,kernel_filter=kernel_filter)
_,out1flipped,_,_=PixelHopPP_Unit(flipped_raw_images,num_kernels=18,saab=saab,window=5,stride=1,train=False,ch_decoupling=False,kernel_filter=kernel_filter)
out1ave=MaxPooling(out1)
out1aveflipped=MaxPooling(out1flipped)
saab,transformed,kernel_filter,falttened=PixelHopPP_Unit(out1ave[0:numoftrainpixelhop],num_kernels=13,saab=None,window=5,stride=1,train=True,energy_th=energy_th,ch_decoupling=True,ch_energies=falttened,kernel_filter=[])
_,out2,_,_=PixelHopPP_Unit(out1ave,num_kernels=13,saab=saab,window=5,stride=1,train=False,ch_decoupling=True,kernel_filter=kernel_filter)
_,out2flipped,_,_=PixelHopPP_Unit(out1aveflipped,num_kernels=13,saab=saab,window=5,stride=1,train=False,ch_decoupling=True,kernel_filter=kernel_filter)
out2ave=MaxPooling(out2)
out2aveflipped=MaxPooling(out2flipped)
saab,transformed,kernel_filter,falttened=PixelHopPP_Unit(out2ave[0:numoftrainpixelhop],num_kernels=11,saab=None,window=5,stride=1,train=True,energy_th=energy_th,ch_decoupling=True,ch_energies=falttened,kernel_filter=[])
_,out3,_,_=PixelHopPP_Unit(out2ave,num_kernels=11,saab=saab,window=5,stride=1,train=False,ch_decoupling=True,kernel_filter=kernel_filter)
_,out3flipped,_,_=PixelHopPP_Unit(out2aveflipped,num_kernels=11,saab=saab,window=5,stride=1,train=False,ch_decoupling=True,kernel_filter=kernel_filter)

size=out3.shape[3]
feature=np.zeros((out3.shape[0],size))
flipped_feature=np.zeros((out3flipped.shape[0],size))
for i in range (out3.shape[0]):
    feature[i,:]=out3[i,0,0,:]
    flipped_feature[i,:]=out3flipped[i,0,0,:]
print(feature.shape)

with open('pairs.txt', 'r') as csvfile:
        trainrows = list(csv.reader(csvfile, delimiter='\t'))[1:600*9+1]
with open('pairs.txt', 'r') as csvfile:
        testrows = list(csv.reader(csvfile, delimiter='\t'))[5401:6000+1]

trainData=[]
trainLabel=[]

for row in trainrows:#testrows
    if(len(row)==3):
        name1=row[0]+'_'+format(int(row[1]), '04d')
        name2=row[0]+'_'+format(int(row[2]), '04d')
        label=1
    elif(len(row)==4):
        name1=row[0]+'_'+format(int(row[1]), '04d')
        name2=row[2]+'_'+format(int(row[3]), '04d')
        label=0#not the same
    flag=0
    for i in range (len(raw_labels)):
        if(raw_labels[i]==name1):
            vect1=np.zeros((1,size))
            vect3=np.zeros((1,size))
            vect1[0]=feature[i,:]
            vect3[0]=flipped_feature[i,:]
            flag+=1
        if(raw_labels[i]==name2):
            vect2=np.zeros((1,size))
            vect4=np.zeros((1,size))
            vect2[0]=feature[i,:]
            vect4[0]=flipped_feature[i,:]
            flag+=1
    if(flag!=2):
        print('Train Error')
        continue
    trainData.append((np.concatenate((sklearn.metrics.pairwise.cosine_similarity(vect1,vect2),np.linalg.norm(vect1-vect2,ord='nuc').reshape(1,1),vect1,vect2),axis=1)))
    trainData.append((np.concatenate((sklearn.metrics.pairwise.cosine_similarity(vect2,vect1),np.linalg.norm(vect2-vect1,ord='nuc').reshape(1,1),vect2,vect1),axis=1)))
    trainData.append((np.concatenate((sklearn.metrics.pairwise.cosine_similarity(vect3,vect4),np.linalg.norm(vect3-vect4,ord='nuc').reshape(1,1),vect3,vect4),axis=1)))
    trainData.append((np.concatenate((sklearn.metrics.pairwise.cosine_similarity(vect4,vect3),np.linalg.norm(vect4-vect3,ord='nuc').reshape(1,1),vect4,vect3),axis=1)))
    trainLabel.append(label)
    trainLabel.append(label)
    trainLabel.append(label)
    trainLabel.append(label)

testData=[]
testLabel=[]
for row in testrows:#testrows
    if(len(row)==3):
        name1=row[0]+'_'+format(int(row[1]), '04d')
        name2=row[0]+'_'+format(int(row[2]), '04d')
        label=1
    elif(len(row)==4):
        name1=row[0]+'_'+format(int(row[1]), '04d')
        name2=row[2]+'_'+format(int(row[3]), '04d')
        label=0#not the same
    #print( name1+' '+name2)
    flag=0
    for i in range (len(raw_labels)):
        if(raw_labels[i]==name1):
            vect1=np.zeros((1,size))
            vect1[0]=feature[i,:]
            flag+=1
            #print(raw_labels[i])
        if(raw_labels[i]==name2):
            vect2=np.zeros((1,size))
            vect2[0]=feature[i,:]
            flag+=1
            #print(raw_labels[i])
    if(flag!=2):
        print('Test Error')
        continue
    testData.append((np.concatenate((sklearn.metrics.pairwise.cosine_similarity(vect1,vect2),np.linalg.norm(vect1-vect2,ord='nuc').reshape(1,1),vect1,vect2),axis=1)))
    testLabel.append(label)

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

trainData=np.asarray(trainData).reshape(-1,2*(size)+2)
trainLabel=np.asarray(trainLabel)
testData=np.asarray(testData).reshape(-1,2*(size)+2)
testLabel=np.asarray(testLabel)

print('trainData.shape',trainData.shape)
print('testData.shape',testData.shape)

clf = SVC(gamma='auto', probability=True)
clf.fit(trainData,trainLabel)
prediction = clf.predict(trainData)
acc1 = accuracy_score(trainLabel, prediction)
print('train acc for energy',energy_th,acc1)
prediction = clf.predict(testData)
acc2 = accuracy_score(testLabel, prediction)
print('test acc for energy',energy_th,acc2)


