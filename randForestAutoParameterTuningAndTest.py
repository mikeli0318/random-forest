"""
    function:
    1. k-fold cross validation
    2. automatically fit the better parameter based on cross-validation
    """
__author__ = "Shiyi Li"

import numpy as np
import math
import RandomTree as rt
import RandomForest as rf#bag learner

def run(leafsize=5,bag=10,verbose=True):
    inf = open("./Yelpdata/data.csv")
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    # compute how much of the data is training and testing
    k_length = int(math.floor(0.2 * data.shape[0])) #5-fold
    train1=data[k_length:]
    test1=data[:k_length]
    train2=np.append(data[:k_length], data[2*k_length:], axis=0)
    test2=data[k_length:2*k_length]
    train3=np.append(data[:2*k_length], data[3*k_length:], axis=0)
    test3=data[2*k_length:3*k_length]
    train4=np.append(data[:3*k_length], data[4*k_length:], axis=0)
    test4=data[3*k_length:4*k_length]
    train5=data[0:4*k_length]
    test5=data[4*k_length:]

    train=[]
    test=[]
    train.append(train1)
    train.append(train2)
    train.append(train3)
    train.append(train4)
    train.append(train5)
    test.append(test1)
    test.append(test2)
    test.append(test3)
    test.append(test4)
    test.append(test5)

    inSamRMSERT=0;
    outSamRMSERT=0;
    inSamRMSElrl = 0;
    outSamRMSElrl = 0;

    with open("Random_forest_evaluation.txt", "w") as f:
        for i in range(5):
            traindata = train[i];
            testdata = test[i];
            trainX = traindata[:, 0:-1]
            trainY = traindata[:, -1]
            testX = testdata[:, 0:-1]
            testY = testdata[:, -1]
            if verbose:
                f.write("cross validation: part"+str(i+1)+"/5\n");
            learner = rf.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leafsize}, bags=bag, boost=False,
                                    verbose=False)
            learner.addEvidence(trainX, trainY)
            predY = learner.query(trainX)
            # evaluate in sample
            # predY = learner.query(trainX) # get the predictions
            rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
            inSamRMSERT = inSamRMSERT + rmse;

            # evaluate out of sample
            predY = learner.query(testX)  # get the predictions
            rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
            outSamRMSERT = outSamRMSERT + rmse;

            rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
            inSamRMSElrl = inSamRMSElrl + rmse;

            predY = learner.query(testX)  # get the predictions
            rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
            outSamRMSElrl = outSamRMSElrl + rmse;
        inSamRMSERT = inSamRMSERT / 5;
        outSamRMSERT = outSamRMSERT / 5;

        inSamRMSElrl = inSamRMSElrl / 5;
        outSamRMSElrl = outSamRMSElrl / 5;
        f.write("======================\n")
        f.write("END OF CROSS VALIDATION\n")
        f.write("Random forest:\n")
        f.write("in-sample rmse:  " + str(inSamRMSERT) + "\n")
        f.write("out-sample rmse:  " + str(outSamRMSERT) + "\n")
        f.write("Linear regression:\n")
        f.write("in-sample rmse:  " + str(inSamRMSElrl) + "\n")
        f.write("out-sample rmse:  " + str(outSamRMSElrl) + "\n")





    return outSamRMSERT



def findParameter():

    inf = open("data.csv")
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

    train_rows = math.floor(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:int(train_rows), 0:-1]
    trainY = data[:int(train_rows), -1]
    testX = data[int(train_rows):, 0:-1]
    testY = data[int(train_rows):, -1]

    # create a learner and train it
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    # learner.addEvidence(trainX, trainY) # train it
    minRMSE=100000;
    minRMSELEAF=0;
    minRMSEBAG=0;
    for bagNum in range(5, 20, 5):
        for leafSize in range(5, 20, 2):

            print "----------------------------"
            print "now working on RF with bag num:"
            print bagNum;
            print "leaf size:"
            print leafSize;
            """
            learner = rf.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leafSize}, bags=bagNum, boost=False, verbose=False)
            learner.addEvidence(trainX, trainY)
            predY = learner.query(testX)  # get the predictions
            rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
            """
            rmse=run(leafsize=leafSize,bag=bagNum,verbose=False)
            if rmse<minRMSE:
                minRMSE=rmse;
                minRMSELEAF = leafSize;
                minRMSEBAG = bagNum;
            print "the best one so far is:"
            print "bag num:"
            print minRMSEBAG;
            print "leaf size:"
            print minRMSELEAF;
            print "min out-sample RMSE"
            print minRMSE




if __name__=="__main__":
    #findParameter();
    run(17,15);#(leaf,bag)
    pass;