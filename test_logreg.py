 
from pdf2image import convert_from_path, convert_from_bytes
import tempfile
from scipy import misc
import numpy as np
import classification_algorithms as ca
import logisticregresssion as lg
import monogramclassification as mc

numClasses=2 #number of classes to predict
#path: path to folder containing training images
#n: dimension of a single image (width x height)
def loadTrainImages(path, n,imagesPerFolder):
    
    trainData=np.zeros(shape=(numClasses,imagesPerFolder,n))
    for i in range(numClasses):
        temp=np.zeros(shape=(imagesPerFolder,dim))
        for j in range(imagesPerFolder):
            image=misc.imread(path+str(i)+"/train"+str(i)+"_"+str(j)+".png")
            temp[j]=np.asarray(image).reshape(-1)
        
        trainData[i]=temp
    return trainData
 
def loadTestImages(path, n,dim):
    trainData=np.zeros(shape=(n,dim))
    for i in range(n):
        image=misc.imread(path+"/monogram"+str(i)+".png")
        trainData[i]=np.asarray(image).reshape(-1)
    return trainData
 

    
imagepath="/Users/kashefkarim/desktop/BA/images/train/"
imagesPerFolder=4 #images per folder in training set
dim=533*533*4 #image dimension
trainimages=loadTrainImages(imagepath,dim, imagesPerFolder)

print(trainimages[0])

adjustedX=trainimages.reshape(imagesPerFolder*numClasses,dim)
#print("Train shape:" ,trainimages.shape)

#print("adjusted shape: ", adjustedX.shape)
testY=np.zeros(shape=(imagesPerFolder*numClasses))

#Crate target vector Y
def setY(testY):
    val=-1
    for i in range(len(testY)):
        if i%imagesPerFolder==0:
            val=val+1
        testY[i]=val


setY(testY)
print("Manual labeling:",testY)


#Run regression

def runClassification(trainX,traint):
#Loading data

#Building design matrix

    X=ca.buildLinearDesignMatrix(trainX)
#calculation weights
    model=ca.multi_logistic_regression(trainX,traint,numClasses)
    
    right_classifications=0
    wrong_classifications=0
    for i in range(len(trainX)):
        (assignment,prob)=model.predicted_label(trainX[i])
    #print("PROB: ",prob)
        if prob>1:
        #print("Probability greater 1!",prob)
            prob=1
        if (assignment==0):
            print("Klasse 0")
            col=color=(prob,0,0)
        elif (assignment==1):
            print("Klasse 1")
            col=color=(0,prob,0)
        elif (assignment==2):
            col=color=(0,0,prob)
    
        
        if traint[i]!=assignment:
            wrong_classifications+=1
        else:
            right_classifications+=1
        
    print("Correct Classifications: ", right_classifications)
    print("Wrong Classifications: ",wrong_classifications)
    print("Correct rate: ", right_classifications/(wrong_classifications+right_classifications))
    
def substract_classifier(p, targets, label):

    for target in targets:  #collection of trainingsbilder target=vector von collectionen
        sum=0.0     #zählvariable, zähler started bei 0.g
        for i in range(len(p)):
            if target[i]!=0:
                sum+=abs(target[i]-p[i])
        if sum <=9000*255:
            return (label,sum)
   
    return ("NOT "+label,sum)
            
#runClassification(adjustedX,testY)
#Train the model with the training data (which are letters in this case)
model=lg.runClassification(adjustedX,testY)

#Test it for single images
#load testimages (which are complete monograms)
imagepath_testimages="/Users/kashefkarim/desktop/BA/images/pngmonograms/"
dim=533*533*4 #image dimension
testimages=loadTestImages(imagepath_testimages,1200,dim)
#adjusted_test_images=testimages.reshape(imagesPerFolder,dim)

#Build design matrix
X=lg.buildLinearDesignMatrix(testimages)

#Return the likelihoods for each class given a single monogram
print("First 100 images")

'''
for i in range(101):
    #print("Likelihood:",model.predicted_likelihood(X[0]))
    print("monogram"+str(i)+" has Predicted Class:",mc.get_letter(model.predicted_label(X[i]),mc.label2number("A")))
    input("Press enter for next image")
'''

diff_A=[testimages[0],testimages[1],testimages[2],testimages[3],testimages[4],testimages[5],testimages[6],testimages[7],testimages[8],testimages[9],testimages[10],testimages[11],testimages[12],testimages[14],testimages[15],testimages[16],testimages[17],testimages[18],testimages[19],testimages[20],testimages[21],testimages[22],testimages[23],testimages[24],testimages[27],testimages[28],testimages[29],testimages[30],testimages[31],testimages[32],testimages[33],testimages[34],testimages[35],testimages[36],testimages[37],testimages[38],testimages[39],testimages[41],testimages[42],testimages[43],testimages[44],testimages[45],testimages[46],testimages[47],testimages[48],testimages[49],testimages[50],testimages[51],testimages[52],testimages[56],testimages[57],testimages[58],testimages[59],testimages[61],testimages[62],testimages[63],testimages[64],testimages[65],testimages[68],testimages[69],testimages[70],testimages[71],testimages[72],testimages[73],testimages[74],testimages[75],testimages[76],testimages[77],testimages[78],testimages[79],testimages[80],testimages[81],testimages[82],testimages[85],testimages[84],testimages[86],testimages[87],testimages[88],testimages[89],testimages[99]]

diff_E=[testimages[6],testimages[7],testimages[8],testimages[5],testimages[9],testimages[10],testimages[11],testimages[12],testimages[13],testimages[51],testimages[71],testimages[78],testimages[122],testimages[129],testimages[130],testimages[164],testimages[183],testimages[209],testimages[210],testimages[211],testimages[213],testimages[215],testimages[222],testimages[220],testimages[223],testimages[224],testimages[225],testimages[230],testimages[231],testimages[239],testimages[240],testimages[241],testimages[242],testimages[244],testimages[245],testimages[249],testimages[251],testimages[259],testimages[263],testimages[265],testimages[270],testimages[271],testimages[289],testimages[319],testimages[318],testimages[344],testimages[345],testimages[353],testimages[357],testimages[372],testimages[375],testimages[395],testimages[454],testimages[466],testimages[493],testimages[495],testimages[496],testimages[510],testimages[511],testimages[551],testimages[582],testimages[590],testimages[596],testimages[602],testimages[604],testimages[682],testimages[701],testimages[702],testimages[712],testimages[725],testimages[729],testimages[753],testimages[762],testimages[865],testimages[972],testimages[975],testimages[982]]

'''for i in range(1,1100):
    print("monogram"+str(i)+" has Predicted Class:",substract_classifier(testimages[i],diff_E,"E"))'''
for i in range(96,105):
    print("monogram"+str(i)+" has Predicted Class:",substract_classifier(testimages[i],diff_A,"A"))
    input("Press enter for next image")
