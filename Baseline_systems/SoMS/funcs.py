
def get_files_in_path(path, ext="wav"):
    '''
    # get files in a path
    -> exp: files = get_files_in_path("./audioFiles")
    '''
    import os, glob
    path = os.path.join(path, "*."+ext)
    theFiles = glob.glob(path, recursive=True)
    return theFiles

def resample_files(path, newPath, newSr):
    '''
    # resample audio files 
    -> exp: resample_files("./audioFiles", "newFiles", 8000)
    '''
    import os, sys, librosa
    filesPaths = get_files_in_path(path)
    if not os.path.exists(newPath): os.makedirs(newPath)
    filesCount = len(filesPaths)
    for i, theFile in enumerate(filesPaths):
        sys.stdout.write("\rresampling files %d%%" % int(100*i/(filesCount)))
        newSig, rate = librosa.load(theFile, sr=newSr)
        newSig.astype(int)
        lastSlashPos = theFile.rfind(os.path.split(theFile)[-1])-1
        # librosa.output.write_wav(newPath+theFile[lastSlashPos:], newSig, newSr, norm=False)
        import scipy.io.wavfile
        scipy.io.wavfile.write(newPath+theFile[lastSlashPos:], newSr, newSig)
    sys.stdout.write("\rresampling all files completed.\n")    

def search_csv(csv_file, search_term, colomn_searched, colomn_out):
    '''
    # search a string in a csv file and a colomn and get it's corresponding value for a different colomn. 
    -> exp: valenz = search_csv('labels-sorted.csv', '001_01.wav', 'Laufnummer', 'Valenz')
    '''
    import pandas as pd
    df = pd.read_csv(csv_file)
    out = df[df[colomn_searched] == search_term][colomn_out]
    ret = out.values
    if len(ret) == 1:
        return ret[0]
    else:
        return -1

def classify_lmh(in_num, out_type=0):
    '''
    # turn number (0-10) into 3 categories (0-2) or lmh(low-medium-high).
    # if out_type == 1 then output is integer 0-2
    -> exp: valenzLMH = classify_lmh(valenz, out_type=1)
    '''
    if in_num <= 4: 
        out = 0
        if not out_type == 0 : out = 'l'
    elif in_num <= 7:
        out = 1
        if not out_type == 0 : out = 'm'
    else:
        out = 2
        if not out_type == 0 : out = 'h'
    return out

def divide_list(list, perc=0.5):
    '''
    # divide a list into two new lists. perc is the first list's share. If perc=0.6 then the first new list will have 60 percent of the original list.
    -> exp: f,s = divide_list([1,2,3,4,5,6,7], perc=0.7)
    '''
    origLen = len(list)
    lim = int(perc*origLen)
    firstList = list[:lim]
    secondList = list[lim:]
    return firstList, secondList
    
def plot_confusion(confusion, actualClasses, PredictedClasses, title):
    '''
    # plot a simple confusion matrix based on matplotlib
    -> exp: plot_confusion(confusion, ["actual low","actual medium","actual high"], ["predicted low","predicted medium","predicted high"], "Self-Assessed")
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    im = ax.imshow(confusion)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(PredictedClasses)))
    ax.set_yticks(np.arange(len(actualClasses)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(PredictedClasses)
    ax.set_yticklabels(actualClasses)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(actualClasses)):
        for j in range(len(PredictedClasses)):
            text = ax.text(j, i, int(confusion[i, j]),
                        ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

def UAR_from_ConfMat(confusion):
    '''
    ### compute UAR (unwaighted averagre recall) from the confusion matrix
    '''
    import numpy as np
    aux = []
    for i, row in enumerate(confusion):
        if sum(row) == 0: continue
        aux.append(row[i]/sum(row))
    uar = np.mean(aux)
    return uar

def writeLineToCSV(csvPath, headers, values):
    '''
    # Write one line to CSV
    -> exp: writeLineToCSV("test.csv", ["a", "b", "c"], ["something",16,34])
    '''
    import pandas as pd
    import os
    dic = {}
    for i, header in enumerate(headers): dic[header] = values[i]
    data = [dic]
    if os.path.exists(csvPath): 
        df = pd.read_csv(csvPath)
        df = df.append(data, ignore_index=True, sort=False)
    else:
        df = pd.DataFrame(data, columns = headers)
    df.to_csv(csvPath, index=False)
