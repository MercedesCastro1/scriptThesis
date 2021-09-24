import os.path as op
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import subprocess

import matplotlib.pyplot as plt



def parseDotBracket(filename):
    """
    Takes as input a dot bracket filename (.b file) and returns an array of "(", ")" and "."
    1. open .b file
    2. split on '\n'
    3. return splitted[1] 
    """
    file = open(filename,"r")
    content = file.read()
    onlyDotBracket = content.split("\n")[1]
    discardTail = onlyDotBracket.split('\t')[0]
    dotBracketList = list(discardTail)
    return dotBracketList

def nodeDegree(chr, i,n):
    base = 2
    if(i == 0 or i == n-1):
        base = 1
    if(chr == '(' or chr == ')'):
        return base + 1
    else:
        return base

def dotBracketToGraph(dotBracketList,index):
    n = len(dotBracketList)
    edges = []
    features = {}
    stack = []
    for i in range(n-1):
       edges.append([i,i+1])
       chr = dotBracketList[i]
       if(chr == '('):
           stack.append(i)
       elif(chr == ')'):
           edges.append([stack.pop(),i])
       features[str(i)] = nodeDegree(chr, i,n)

    if(dotBracketList[n-1] == ')'):
        edges.append([stack.pop(),n-1])
    features[str(n-1)] = nodeDegree(chr, n-1,n)

    return {"edges": edges, "features": features}

def candidatesDatasetToGraph(candidates,outputPath):
    import json
    index = 0
    for dotBracket in candidates:
        graph = dotBracketToGraph(dotBracket,index)
        js = json.dumps(graph)
        fname = outputPath  + str(index) + ".json"
        f = open(fname, "w")
        f.write(js)
        f.close()
        index += 1

def graphTovec(input_directory, output_file):
    print("Graph2Vec",input_directory,output_file)
    subprocess.run("python3 graph2vec/src/graph2vec.py --input-path " + input_directory +" --output-path " + output_file, text=True, shell = True)


def kmeansGraph(csv_path, output_path):
    data = pd.read_csv(csv_path)
    for k in range(2,7):    
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        fname = output_path + str(k) + ".csv"
        f = open(fname, "w")
        for i, label in enumerate(kmeans.labels_):
            f.write(",".join([str(i),str(label)]) + "\n")
        f.close()





def optimalK(data, nrefs=3, maxClusters=5):
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
        refDisps = np.zeros(nrefs)
        for i in range(nrefs):
            
            randomReference = np.random.random_sample(size=data.shape)
            
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    return (gaps.argmax() + 1, resultsdf)

# csv_data = pd.read_csv("/Users/mercedes/Documents/IRB/project/code/output_csv/w_0_180.csv")
# score_g, df = optimalK(csv_data, nrefs=5, maxClusters=8)
# print("score_g", score_g)
# print("df", df)
# plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
# plt.xlabel('K');
# plt.ylabel('Gap Statistic');
# plt.title('Gap Statistic vs. K');


def elbowMethodKmeans(data):
    from yellowbrick.cluster import KElbowVisualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
    visualizer.fit(data)
    visualizer.show()

#elbowMethodKmeans(csv_data)







def parseFastaFile(fastaFile):
    """ returns the the rna sequence"""
    f = open(fastaFile)
    content = f.read()
    return content.split("\n")[1]

def splitIntoWindows(fastaSeq, wSize, overlap, outputDir):
    from pathlib import Path
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    i = 0
    fileBase = outputDir+"/w_" 
    result = []
    while i < len(fastaSeq) - wSize:
        window = fastaSeq[i: i+ wSize]
        filename = fileBase + str(i) + "_" + str(i+wSize)
        f = open(filename,"w")
        f.write("".join(window))
        f.close()
        result.append(filename)
        i += wSize-overlap
    if i < len(fastaSeq):
        window = fastaSeq[i:]
        filename = fileBase + str(i) + "_" + str(len(fastaSeq))
        f = open(filename,"w")
        f.write(window)
        f.close()
        result.append(filename)
    return result



def execViennaRNA(filename, instances):
    proc = subprocess.run("RNAsubopt -p " + str(instances) + " < "+ filename, capture_output=True, text=True, shell = True)
    out = proc.stdout
    return out.split("\n")[:-1]


def matchBracketWithNextPars(dotBracketChar, nextParsScore, upper_threshold, lower_threshold):
    if(dotBracketChar == '.'):
        return (nextParsScore < lower_threshold)
    else:
        return (nextParsScore >= upper_threshold)
       
#print(matchBracketWithNextPars('(',0.1,0.3))

def nextPARSfilter(dotBracket, nextParsSeq, upper_threshold, lower_threshold):
    """
    dotbracket = ["(",".", ......] of length n
    nextParsSeq  ["0.1","0.8", ..] of length n
    
    accumError = 0
    For(i= 0 , i < n , i++) {
        if(matchBracketWithNextPars(dotbracket[i], nextParsSeq[i],0.3))
          error++
    }
    return accumError

    """
    for i in range(len(dotBracket)-1):
        if not(matchBracketWithNextPars(dotBracket[i], nextParsSeq[i], upper_threshold, lower_threshold)):
            return False
    
    return True

def nextPARSfilter_v2(dotBracket, nextParsSeq, upper_threshold, lower_threshold, tolerance):
    
    count_rejected_bases = 0 
    for i in range(len(dotBracket)-1):
        if not(matchBracketWithNextPars(dotBracket[i], nextParsSeq[i], upper_threshold, lower_threshold)):
            if(nextParsSeq[i] != 0.5):
                count_rejected_bases = count_rejected_bases + 1
    if count_rejected_bases / len(dotBracket) > tolerance:
        return False

    return True



def filterCandidates(dotBracketCandidates, nextParsScore):
    result = []
    for dotBracket in dotBracketCandidates:
        if nextPARSfilter_v2(dotBracket, nextParsScore, upper_threshold = 0.5, lower_threshold = 0.7, tolerance = 0.5):
            result.append(dotBracket)

    return result

def saveDotBracketStructures(dotBracketCandidates, path):
    index = 0
    for dotBracket in dotBracketCandidates:
        fname = path  + str(index) + ".b"
        f = open(fname, "w")
        f.write(dotBracket)
        f.close()
        index += 1

def nextParsScores(nextParseFile):
    f = open(nextParseFile)
    content = f.read()
    nextParsScoresStrings = content.strip().split(";")[1:-1]
    nextParsScores = list(map(lambda x: (float(x)+1)/2,nextParsScoresStrings))
    return nextParsScores

def nextParsWindow(nextPARS_scores,window_name):
    splitted = window_name.split("/")[-1].split(".")[0].split("_")
    start , end = int(splitted[1]), int(splitted[2])
    return nextPARS_scores[start:end]

#filterCandidates([ "/Users/mercedes/Documents/IRB/project/data/b_files/21Aug02.b/21Aug02-18-03-17_1.b", "/Users/mercedes/Documents/IRB/project/data/b_files/21Aug02.b/21Aug02-18-03-17_2.b", "/Users/mercedes/Documents/IRB/project/data/b_files/21Aug02.b/21Aug02-18-03-17_3.b"], nextParsScores)


def removeDir(dir_path):
    subprocess.run("rm -rf "+ dir_path, shell = True)

def createDir(dir_path):
    subprocess.run("mkdir -p "+ dir_path, shell = True)

    

def createOutputDirs(base_dir):
    createDir(base_dir)
    createDir(base_dir + "windows")
    createDir(base_dir + "dot_bracket")
    createDir(base_dir + "cluster")
    createDir(base_dir + "graphs")
    createDir(base_dir + "csv")




def create_drawing_file(window_path, structures_dir, cluster_path, n_clust):
    fw = open(window_path, "r")
    seq = fw.read()
    fw.close()
    fa_files = []
    for i in range(n_clust):
        fa_files.append(open("clust_" + str(i) + ".fa", "w"))
    fc = open(cluster_path, "r")
    fc_lines = fc.read().strip().split("\n")
    fc.close()
    for line in fc_lines:
        struc_index, cluster = line.split(",")[0], int(line.split(",")[1])
        fs = open(structures_dir + "/" + struc_index + ".b" , "r")
        structure = fs.read()
        fs.close()
        fa_files[cluster].write(">C1_00080C_A_" + struc_index + "\n" + seq + "\n" + structure + "\n")
    
    for f in fa_files:
        f.close()
    



def processFasta(fastaFile, next_pars_path):
    fastaSeq = parseFastaFile(fastaFile)
    baseDir = "output/" + fastaFile.split("/")[-1].split(".")[0] + "/"
    removeDir(baseDir)
    createOutputDirs(baseDir)
    windowFiles = splitIntoWindows(fastaSeq, 180,60, baseDir + "windows")
    nextPARScores = nextParsScores(next_pars_path)
    for f in windowFiles:
        window_name = f.split("/")[-1]
        nextPARSW = nextParsWindow(nextPARScores,f)
        dotBracketCandidates = execViennaRNA(f, 1000)
        refinedCandidates = filterCandidates(dotBracketCandidates,nextPARSW)
        if len(refinedCandidates) > 4:

            dot_bracket_dirname = baseDir + "dot_bracket/" + window_name + "/"
            createDir(dot_bracket_dirname)
            saveDotBracketStructures(refinedCandidates, dot_bracket_dirname)

            graph_dirname = baseDir + "graphs/" + window_name + "/"
            createDir(graph_dirname)
            candidatesDatasetToGraph(refinedCandidates,graph_dirname)

            csv_path = baseDir + "csv/"+window_name+".csv"
            graphTovec(graph_dirname, csv_path)

            cluster_dirname = baseDir + "cluster/" + window_name + "/"
            createDir(cluster_dirname)
            kmeansGraph(csv_path, cluster_dirname)
    




processFasta("/Users/mercedes/Documents/IRB/project/data/fasta/spike_ins.fasta","/Users/mercedes/Documents/IRB/project/data/score_files/BLACAT1_1.csv")

#Graph2Vec output/C1_00080C_A/graphs/w_0_180/ output/C1_00080C_A/csv/w_0_180.csv
        

create_drawing_file("output/spike_ins/windows/w_0_180", "output/spike_ins/dot_bracket/w_0_180", "output/spike_ins/cluster/w_0_180/4.csv",4)