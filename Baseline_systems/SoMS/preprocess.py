import os, sys
import pandas as pd
import numpy as np
from funcs import *

def main():
    filesPath = "/run/media/sina/207AB4977AB46AE4/datasets/AVEC2019/"
    trainCSVpath = os.path.join(filesPath,"Train_Scores","TRAIN_SOM_AVEC2019.csv")
    devCSVpath = os.path.join(filesPath,"Development_Scores","DEVELOPMENT_SOM_AVEC2019.csv")
    testCSVpath = os.path.join(filesPath,"Test_Scores","TEST_SOM_AVEC2019.csv")
    trainMetaCSVpath = os.path.join(filesPath,"Train_Scores","Train_Meta_Data.csv")
    devMetaCSVpath = os.path.join(filesPath,"Development_Scores","Development_Meta_Data.csv")
    testMetaCSVpath = os.path.join(filesPath,"Test_Scores","test_meta.csv")

    trainCSVpathDynamic = os.path.join(".","data","TRAIN_SOM_AVEC2019_Dynamic.csv")
    makeDynamicCSV(trainCSVpath, trainMetaCSVpath, trainCSVpathDynamic, origCSVCol='File', metaCSVCol='Participant')
   
    devCSVpathDynamic = os.path.join(".","data","DEVELOPMENT_SOM_AVEC2019_Dynamic.csv")
    makeDynamicCSV(devCSVpath, devMetaCSVpath, devCSVpathDynamic, origCSVCol='File', metaCSVCol='Participant')

    testCSVpathDynamic = os.path.join(".","data","TEST_SOM_AVEC2019_Dynamic.csv")
    makeDynamicCSV(testCSVpath, testMetaCSVpath, testCSVpathDynamic, origCSVCol='File', metaCSVCol='Participant')

    trainCSVpathDynamicEqual = os.path.join(".","data","TRAIN_SOM_AVEC2019_Dynamic_Equal.csv")
    classFunc = lambda y: int(y) - 1
    equalizeCSVbasedOnClasses(trainCSVpathDynamic, trainCSVpathDynamicEqual, colomnTarget="Valance", classFunc=classFunc)

def makeDynamicCSV(origCSVpath, metaCSVpath, newCSVpath, origCSVCol='File', metaCSVCol='Participant'):
    df1 = pd.read_csv(origCSVpath)
    df2 = pd.read_csv(origCSVpath)
    my_list = df1["File"].values
    lastSlashPos = newCSVpath.rfind(os.path.split(newCSVpath)[-1])-1
    savePath = newCSVpath[:lastSlashPos]
    if not os.path.exists(savePath): os.makedirs(savePath)
    if os.path.exists(newCSVpath): os.remove(newCSVpath)
    for i, fileName in enumerate(my_list):
        sys.stdout.write("\rmaking dynamic files %d%%" % int(100*i/(len(my_list))))
        Valance = search_csv(origCSVpath, fileName, origCSVCol, 'Valance')
        Arousal = search_csv(origCSVpath, fileName, origCSVCol, 'Arousal')
        Valence0 = search_csv(metaCSVpath, fileName[:-3], metaCSVCol, 'Valence_0')
        Arousal0 = search_csv(metaCSVpath, fileName[:-3], metaCSVCol, 'Arousal_0')
        writeLineToCSV(newCSVpath, ["File", "Valance", "Arousal", 'Valence_0', 'Arousal_0'], [fileName,Valance,Arousal,Valence0,Arousal0])
    sys.stdout.write("\rmaking dynamic files completed and saved to %s\n" % newCSVpath)

def equalizeCSVbasedOnClasses(origCSVpath, newCSVpath, colomnTarget="Valance", classFunc=lambda y: y):
    fileNames = []
    labels = []
    labelsFunced = []
    df = pd.read_csv(origCSVpath)
    my_list = df["File"].values
    filesCount = len(my_list)
    for i, fileName in enumerate(my_list):
        sys.stdout.write("\rpreparing files %d%%" % int(100*i/(filesCount)))
        Target = df[i:i+1][colomnTarget]
        fileNames.append(fileName)
        labels.append(Target)
        labelsFunced.append(classFunc(Target))
    sys.stdout.write("\rpreparing files completed\n")
    # print(labelsFunced.count(0),labelsFunced.count(1),labelsFunced.count(2))
    maxItem = max(set(labelsFunced), key=labelsFunced.count)
    maxNum = labelsFunced.count(maxItem)
    # print(maxNum, maxItem)
    # print(maxNum, labelsFunced.count, set(labelsFunced))
    for item in set(labelsFunced):
        if item == maxItem: continue
        diff = maxNum - labelsFunced.count(item)
        if diff <= 0: continue
        indices = [i for i, x in enumerate(labelsFunced) if x == item]
        samples = np.random.choice(indices, diff, replace=True)
        # print(item, samples, len(labelsFunced))
        for j, sample in enumerate(samples):
            sys.stdout.write("\rduplicating files refrences for label %d %d%%" % (item, int(100*j/(len(samples)))))
            # data = {'File': fileNames[sample], 'Valance': int(labels[sample])}
            # df = df.append(data, ignore_index=True)
            df = df.append(df.loc[sample], ignore_index=True)
            # writeLineToCSV(newCSVpath, df.columns.values, df.loc[sample].values)
        sys.stdout.write("\rduplicating files refrences completed for label %d\n" % item)
    df.to_csv(newCSVpath, index=False)

def filterClasses(origCSVpath, newCSVpath, colomnTarget="Valance", includeList=[]):
    fileNames = []
    labels = []
    df = pd.read_csv(origCSVpath)
    my_list = df["File"].values
    filesCount = len(my_list)
    for i, fileName in enumerate(my_list):
        sys.stdout.write("\rpreparing files %d%%" % int(100*i/(filesCount)))
        Target = df[i:i+1][colomnTarget]
        fileNames.append(fileName)
        labels.append(Target)
    sys.stdout.write("\rpreparing files completed\n")
    data = []
    for i, label in enumerate(labels):
        if not int(label) in includeList: continue
        data.append(df.loc[i])
    df = pd.DataFrame(data)
    df.to_csv(newCSVpath, index=False)

if __name__ == "__main__":
    main()