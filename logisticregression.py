#First, we load the data we'd like to train

import numpy as np
from random import random



#Secondly, we build a logistic classifier
class logistic_regression:
    def __init__(self,X,t,threshold):
        self.X=X
        self.t=t
        self.threshold=threshold
        self.w=np.random.rand(len(X[0]))/1000000 #randomly drawn weights
        self.calculateW()
        
    def calculateW(self):
        sigmoid=lambda x: 1/(1+np.exp(-x))
        
        def Err():
            sum=0
            for i in range(len(self.t)):
              # print(self.w.T.dot(self.X[i]))
                sum=sum+(sigmoid(self.w.T.dot(self.X[i]))-self.t[i])*(self.X[i].T)
            return sum
        iter=0
        alpha=0.0000000001
        while iter<50:
            self.w=self.w-alpha*Err()
           # print("Max w:",max(self.w))
            iter=iter+1
            print("Training "+str(100*iter/50)+"% complete")

    def predicted_label(self,X):
        sigmoid=lambda x: 1/(1+np.exp(-x))
        likelihood=sigmoid(self.w.T.dot(X))
        if likelihood>=self.threshold:
            return (1,likelihood)
        else:
            return (0,likelihood)
   
    def predicted_likelihood(self,X):
        sigmoid=lambda x: 1/(1+np.exp(-x))
        likelihood=sigmoid(self.w.T.dot(X))
        return likelihood
    
    def get_weights(self):
        return self.w
            
#In order to use the logistic classifier, we have to transform our original data
#using a design matrix. Caution: This is more complicated as you may think, because
#we have 20 dimensional input data. 
#Because of the curse of dimensionality, it makes sense to reduce alike dimensions
#later on. This function is not correct yet, so do not use.
#d=maximum degree of polynomial
#import sys
#from scipy.special import comb

def buildPolynomialDesignMatrix(data, d):
    m=1 #m=number of basis functions
    
    for i in range(1,d+1):
        m=m+comb(len(data[0])+i-1,i)
    X=np.ones(shape=(len(data),int(m))) #design matrix
    
    def applyPower(x,arr):
        sum=1
        for i in range(len(x)):
            sum*=x[i]**arr[i]
            
        return sum  
    
    iter1=0
    iter2=0
    for D in data: #for each row of the design matrix
        #dynamic for loop to calculate all combinations of the basis functions in a given row
        arr=np.zeros(len(data[0]))
        arr[len(data[0])-1]=1
        for iter2 in range(1,len(data[0])+1):
            X[iter1,iter2]=applyPower(D,arr)
            print(arr)
            #sys.exit()
            #Now we increment the loop. This is equivalent, to adding two numbers of base d 
            overflow=True
            r=len(arr)-1        
            while (overflow and r>=0):
                #increment innermost variable and check if overflow
                arr[r]+=1
                if (arr[r]>d):
                    arr[r]=0
                    r=r-1
                else:
                    overflow=False  
        sys.exit()
        iter1+=1
    return X

#The design matrix in the training phase should be the same
#as in the testing phase.
#If you change this function you have to equally change
#the design matrix function in the "sample.submission.py".
    
def buildLinearDesignMatrix(data):
    X=np.ones(shape=(len(data),len(data[0])+1))
    
    for i in range(len(data)):
        X[i,1:,]=data[i]
    return X
    
#X=buildPolynomialDesignMatrix(trainX,1)




#Beware: Overfitting can happen very quickly, so don't take these
#values too literally
#f=classification model we used
def classificationLoss(m,testX,testt):
    sum=0
    count=0
    for i in range(len(testt)):
        if testt[i]==0 and m.predicted_label(testX[i])==1:
            sum+=1
        elif testt[i]==1 and m.predicted_label(testX[i])==0:
            sum+=5
        else:
            count+=1

    return (count,len(testt)-count,sum/len(testt))

def runClassification(trainX, traint):
    
    #Building design matrix
    X=buildLinearDesignMatrix(trainX)
    #calculation weights
    model=logistic_regression(X,traint)
    
    #estimate error
    (correct,false,err)=classificationLoss(model,buildLinearDesignMatrix(trainX),traint)
    print("Number of false classifications on test set:",false)
    print("Number of correct classifications on test set:",correct)
    print("Estimated classification loss is (lower is better):",err)
    
    #Save the weights in our submission folder,
    np.savetxt("/Users/kashefkarim/desktop/BA/result.txt",model.w)
    return model


