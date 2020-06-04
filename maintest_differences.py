
#from pdf2image import convert_from_path, convert_from_bytes
import tempfile

import numpy as np
import imageio
import monogram_labels as ml

numClasses=2 #number of classes to predict
#path: path to folder containing training images
#n: dimension of a single image (width x height)
def loadTrainImages(path, n,imagesPerFolder):
    
    trainData=np.zeros(shape=(numClasses,imagesPerFolder,n))
    for i in range(numClasses):
        temp=np.zeros(shape=(imagesPerFolder,dim))
        for j in range(imagesPerFolder):
            image=imageio.imread(path+str(i)+"/train"+str(i)+"_"+str(j)+".png")
            temp[j]=np.asarray(image).reshape(-1)
        
        trainData[i]=temp
    return trainData
 
def loadTestImages(path, n,dim,start):
    trainData=np.zeros(shape=(n,dim))
    for i in range(0,n):
        image=imageio.imread(path+"/monogram"+str(start+i)+".png")
        trainData[i]=np.asarray(image).reshape(-1)
    return trainData

def loadSpecificImages(path, dim,image_ids):
    n=len(image_ids)
    trainData=np.zeros(shape=(n,dim))
    for i in range(0,n):
        image=imageio.imread(path+"/monogram"+str(image_ids[i])+".png")
        trainData[i]=np.asarray(image).reshape(-1)
    return trainData
 

  

def substract_classifier(p, targets, label,threshold):

    for target in targets:  #collection of trainingsbilder target=veqctor von collectionen
        sum=0.0     #zaehlvariable, zaehler started bei 0.g
        for i in range(len(p)):
            if target[i]!=0:
                sum+=abs(target[i]-p[i])
        if sum <=threshold:
            return (label,sum,1)
   
    return ("NOT "+label,sum,0)
            
#runClassification(adjustedX,testY)
#Train the model with the training data (which are letters in this case)
#model=lg.runClassification(adjustedX,testY)

#Test it for single images
#load testimages (which are complete monograms)
n_testimages=1000#number of training images
imagepath_testimages="/home/pi/BA/images/pngmonograms/"
dim=533*533*4 #dimension of collection of trainimages

#diff_A=[testimages[0],testimages[1],testimages[2],testimages[3],testimages[4],testimages[5],testimages[6],testimages[7],testimages[8],testimages[9],testimages[10],testimages[11],testimages[12],testimages[14],testimages[15],testimages[16],testimages[17],testimages[18],testimages[19],testimages[20],testimages[21],testimages[22],testimages[23],testimages[24],testimages[27],testimages[28],testimages[29],testimages[30],testimages[31],testimages[32],testimages[33],testimages[34],testimages[35],testimages[36],testimages[37],testimages[38],testimages[39],testimages[41],testimages[42],testimages[43],testimages[44],testimages[45],testimages[46],testimages[47],testimages[48],testimages[49],testimages[50],testimages[51],testimages[52],testimages[56],testimages[57],testimages[58],testimages[59],testimages[61],testimages[62],testimages[63],testimages[64],testimages[65],testimages[68],testimages[69],testimages[70],testimages[71],testimages[72],testimages[73],testimages[74],testimages[75],testimages[76],testimages[77],testimages[78],testimages[79],testimages[80],testimages[81],testimages[82],testimages[85],testimages[84],testimages[86],testimages[87],testimages[88],testimages[89],testimages[99]]
#diff_E=[testimages[6],testimages[7],testimages[8],testimages[5],testimages[9],testimages[10],testimages[11],testimages[12],testimages[13],testimages[51],testimages[71],testimages[78]]
#trainimages=loadSpecificImages(imagepath_testimages,n_testimages,dim,[0,1,2,3,4,5])
#diff_A=loadSpecificImages(imagepath_testimages,dim,[0,1,2,3,4])
diff_E=loadSpecificImages(imagepath_testimages,dim,ml.labels_E)
#diff_E=ml.clear_labels_E
#diff_X=loadSpecificImages(imagepath_testimages,dim,[6,7,8,5,9,10,11])
#diff_P=loadSpecificImages(imagepath_testimages,dim,[6,7,8,5,9,10,11])
#diff_B=loadSpecificImages(imagepath_testimages,dim,[6,7,8,5,9,10,11])

#diff_E=[testimages[6],testimages[7],testimages[8],testimages[5],testimages[9],testimages[10],testimages[11],testimages[12],testimages[13],testimages[51],testimages[71],testimages[78],testimages[122],testimages[129],testimages[130],testimages[164],testimages[183],testimages[209],testimages[210],testimages[211],testimages[213],testimages[215],testimages[222],testimages[220],testimages[223],testimages[224],testimages[225],testimages[230],testimages[231],testimages[239],testimages[240],testimages[241],testimages[242],testimages[244],testimages[245],testimages[249],testimages[251],testimages[259],testimages[263],testimages[265],testimages[270],testimages[271],testimages[289],testimages[319],testimages[318],testimages[344],testimages[345],testimages[353],testimages[357],testimages[372],testimages[375],testimages[395],testimages[454],testimages[466],testimages[493],testimages[495],testimages[496],testimages[510],testimages[511],testimages[551],testimages[582],testimages[590],testimages[596],testimages[602],testimages[604],testimages[682],testimages[701],testimages[702],testimages[712],testimages[725],testimages[729],testimages[753],testimages[762],testimages[865],testimages[972],testimages[975],testimages[982]]

from datetime import datetime
def test_algorithm(train_images,data_labels,letter,threshold):
    true_positives=0
    true_negatives=0
#    positives=len(data_labels)
    incorrect_guesses=0
    chunksize=5
    chunks=int(np.ceil(n_testimages/chunksize))
    #print("Chunks:",chunks)
    now = datetime.now() # current date and time
    date_time = now.strftime("%m_%d_%Y_%H-%M-%S")
    result="Algorithm used: Differences\nTrainingsimages: "+str(len(train_images))+"\nTestimages:"+str(n_testimages)+"\nThreshold: "+str(threshold)+"\n"
    for j in range(chunks):
        startpos=int(j*chunksize)
        testimages=loadTestImages(imagepath_testimages,chunksize,dim,startpos)
        for i in range(0,chunksize):
            prediction=substract_classifier(testimages[i],train_images,letter,threshold)
            output="monogram"+str(i+startpos)+" has Predicted Class:"+prediction[0]+"\n"
            print(output)
            result+=output
            if prediction[2]==1 and (i+startpos) in data_labels:
                true_positives+=1
            elif prediction[2]==0 and (i+startpos) not in data_labels:
                true_negatives+=1
            else:
                incorrect_guesses+=1
#    print("True positive rate:",true_positives/(positives))
  #  print("True negative rate:",true_negatives/negatives)
    print("Accuracy:",(true_positives+true_negatives)/(n_testimages))
  #  result+="True positive rate:"+str(true_positives/(positives))+"\n"
 #   result+="True negative rate:"+str(true_negatives/negatives)+"\n"
    result+="Accuracy:"+str((true_positives+true_negatives)/(n_testimages))+"\n"
    file = open("./../runs/"+date_time+str(n_testimages)+"_"+letter+".txt", 'w')
    file.write(result)
    file.close()

#test_algorithm(diff_E,ml.labels_E,"E",9000*255*0.6)
test_algorithm(diff_E,ml.labels_E,"E",1)
#test_algorithm(diff_E,ml.labels_E,"E",9000*255*0.2)