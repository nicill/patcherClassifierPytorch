import sys
import abc

class resultFileProcesser():
    def __init__(self,fileName1,fileName2):
        self.inFileName=fileName1
        self.outFileName=fileName2
    def run(self):
        #print("Processing File")
        self.f1 = open(self.inFileName, "r")
        self.f2 = open(self.outFileName, "a")
        for line in self.f1:
            self.processLine(line)
        self.flush()

    @abc.abstractmethod
    def  processLine(self,line):
        #" Method to process whatever is done in each experiment, not implemented in the mother class "
        return

    @abc.abstractmethod
    def flush(self):
        "method to be run after all lines have been processed, to be implemented in the subclasses"
        return

class marianoResultFileProcesser(resultFileProcesser):
    SUBSTAGES=6
    def __init__(self,fileName1,fileName2,important):
        #print("model;LR;Unfreeze;Mosaic;Category;DICECM;DICEREF")
        super().__init__(fileName1,fileName2)
        self.patchSize=-1
        self.important=[]
        self.outputDict={}
        for imp in important:
            #print("Running results processor for Mariano Style code, important class found "+imp)
            self.important.append(imp)
            self.outputDict[imp]=[[],[],[],[]]

        self.processHeader()
        self.restart()

    def processHeader(self):
        self.f1 = open(self.inFileName, "r")
        for line in self.f1:
            if "number of read mosaics" in line: self.numMosaics=int(line.strip().split(" ")[4])
            elif "Starting cross-validation" in line:break
        #print("Finished processing header "+str(self.numMosaics))
        self.f1.close()

    def restart(self):
        self.modelName=""
        self.stage=1
        self.substage=0

    def  processLine(self,line):
        #print(line)
        if "starting (" in line:
            if self.patchSize==-1: self.patchSize=int(line.split("(")[1].split(",")[0])
            elif self.patchSize!=int(line.split("(")[1].split(",")[0]):raise Exception("Two patch sizes in same file")
        elif "initializing feature extractor with architecture" in line and self.stage==1:
            aux=line.strip().split(" ")
            model=aux[5]
            lr=float(aux[7])
            frozen=aux[10]
            for x in self.outputDict.keys():
                self.outputDict[x][0].append(model+str(" ")+str(lr)+str(" ")+str(frozen)+" ")
        elif "Starting training for mosaic" in line:
            currentMosaic=int(line.split(" ")[5])
            #print("mosaic found "+str(currentMosaic))
            #if(currentMosaic!=self.stage):raise Exception("Unfinished stage error "+str(currentMosaic)+" "+str(self.stage))
        elif any([item in self.important for item in line.strip().split(" ")]):
            #print(" found important line "+line)
            if "TPR" in line:
                currentImportantLabel=line.split(" ")[0]
                self.outputDict[currentImportantLabel][1].append(float(line.strip().split(" ")[2]))
            if "FPR" in line:
                currentImportantLabel=line.split(" ")[0]
                self.outputDict[currentImportantLabel][2].append(float(line.strip().split(" ")[2]))
            if "ACCURACY" in line:
                currentImportantLabel=line.split(" ")[0]
                self.outputDict[currentImportantLabel][3].append(float(line.strip().split(" ")[2]))
        elif "DICE COEFFICIENT" in line:
            self.substage+=1
            if self.substage==marianoResultFileProcesser.SUBSTAGES:
                #print("Substage reset "+str(self.substage)+" stage is now "+str(self.stage))

                self.substage=0
                if self.stage==self.numMosaics:
                #    print("restarting at stage "+str(self.stage))
                    self.restart()
                else: self.stage+=1

    def flush(self):
        #print(self.outputDict)
        for key,value in self.outputDict.items():
            i=0
            # value contains 4 lists, one with the info of the model, and three with TPR, FPR and accuracy
            while(i<len(value[0])):
                outString=value[0][i]+key+" "

                for k in range(1,4):# because we are storing three things TPR, FPR and accuracy
                    av=0
                    for j in range(self.numMosaics):
                        aux=value[k][self.numMosaics*i+j]
                        av+=aux
                        outString+=str(aux)+" "
                    av=av/self.numMosaics
                    outString+=str(av)+" "
                outString+="\n"

                i+=1

                print(outString,end="")
#        if self.output!="":
#            print(self.output)
#            self.f2.write(self.output+"\n")
#            self.restart()



class segmenterResultFileProcesser(resultFileProcesser):
    NUMBER_STAGES=5 #including 2 for stats
    def __init__(self,fileName1,fileName2):
        print("model;LR;Unfreeze;Mosaic;Category;DICECM;DICEREF")
        super().__init__(fileName1,fileName2)
        self.restart()

    def restart(self):
        self.modelName=""
        self.stage=0
        self.diceList=[]
        self.output=""
    def mosaicLine(self,linePart):return "with prefix" in linePart
    def layerLine(self,linePart):return "LAYER" in linePart
    def modelLine(self,linePart):return "@@@@@@@@@" in linePart
    def coarseLine(self,linePart):return "dice coarse" in linePart
    def fineLine(self,linePart):return "dice refined" in linePart

    def flush(self):
        if self.output!="":
            print(self.output)
            self.f2.write(self.output+"\n")
            self.restart()

    def processLine(self,line):
        # if we find an exception, restart
        linePart=line[0:100]
        #print(line)
        #first, process stage changes
        if "EXCEPTION" in linePart:self.restart()
        elif self.modelLine(linePart):
            self.flush()
            listOfWords=line.split(" ")
            self.output+=listOfWords[7]+";"+listOfWords[4]+";"+listOfWords[10].strip()
        elif self.mosaicLine(linePart):
            listOfWords=line.strip().split(" ")
            self.output+=";"+listOfWords[3]
        elif self.layerLine(linePart):
            listOfWords=line.strip().split(" ")
            self.output+=";"+listOfWords[1]
        elif self.coarseLine(linePart):
            listOfWords=line.strip().split(" ")
            self.output+=";"+listOfWords[3][:6]+";"
        elif self.fineLine(linePart):
            listOfWords=line.strip().split(" ")
            self.output+=listOfWords[3][:6]

class TLresultFileProcesser(resultFileProcesser):
    NUMBER_STAGES=5 #including 2 for stats
    def __init__(self,fileName1,fileName2):
        print("model;LR;Unfreeze;TA;TAFP;PA")
        super().__init__(fileName1,fileName2)
        self.restart()

    def restart(self):
        self.modelName=""
        self.stage=0
        self.TA=0
        self.TAFP=0
        self.PA=0
        self.output=""

    def flush(self):
        pass

    def upStage(self):
        self.stage+=1

    def stageChange(linePart):
        return ("FULL AGREEMENT" in linePart) or ("Partial AGREEMENT" in linePart)

    def statsStage(linePart):
        return "Stats " in linePart

    def isFloat(linePart):
        try:
            a=float(linePart)
            return True
        except ValueError:
            return False
    def lastStage(self): return self.stage==self.NUMBER_STAGES
    def activeStage(self): return not self.stage==0

    def processLine(self,line):
        # if we find an exception, restart
        linePart=line[0:100]
        #print("LINE! "+line)
        #first, process stage changes
        if "EXCEPTION" in linePart:self.restart()
        elif "computing" in linePart:
            listOfWords=line.split(" ")
            #print(str(listOfWords)+" "+str(len(listOfWords)))
            self.output+=listOfWords[7]+";"+listOfWords[4]+";"+listOfWords[10].strip()
        elif TLresultFileProcesser.stageChange(linePart) :
            #print("yelou "+linePart+" "+str(self.stage))
            self.upStage()
        elif TLresultFileProcesser.isFloat(linePart.replace("%"," ").strip()):
            if self.activeStage():self.output+=";"+linePart.replace("%"," ").strip()
        elif TLresultFileProcesser.statsStage(linePart):
            listOfWords=line.strip().split("(")
            category=listOfWords[0].split(" ")[1]
            otherlistOfWords=(listOfWords[1])[0:-1]
            finalString=otherlistOfWords.split(",")

            total=finalString[0].strip()
            TP=finalString[1].strip()
            FP=finalString[2].strip()
            TN=finalString[3].strip()
            FN=finalString[4].strip()

            self.output+=";"+category+";"+total+";"+TP+";"+FP+";"+TN+";"+FN+";"+";"+";"+";"
            self.upStage()
        #check if we need to restart
        if(self.lastStage()):
                print(self.output)
                self.f2.write(self.output+"\n")
                self.restart()

class UnetResultFileProcesser(resultFileProcesser):
    NUMBER_STAGES=7
    def __init__(self,fileName1,fileName2):
        print("LR;deciduous;evergreen")
        super().__init__(fileName1,fileName2)
        self.restart()

    def restart(self):
        self.modelName=""
        self.stage=0
        self.output=""

    def flush(self):
        pass

    def upStage(self):
        self.stage+=1
        #print("UP! "+str(self.stage)+" "+self.output)


    def lastStage(self): return self.stage==self.NUMBER_STAGES

    def processLine(self,line):
        # if we find an exception, restart
        linePart=line[0:100]
        #print("LINE! "+line)
        #first, process stage changes
        if "EXCEPTION" in linePart:self.restart()
        elif "Starting with LR" in linePart:
            listOfWords=line.split(" ")
            #print(str(listOfWords)+" "+str(len(listOfWords)))
            self.output+=listOfWords[3].strip()
        elif "dice Unet decidious" in linePart:
            listOfWords=line.split(" ")
            #print(str(listOfWords)+" "+str(len(listOfWords)))
            self.output+=";"+listOfWords[3].strip()
        elif "dice Unet evergreen" in linePart:
            listOfWords=line.split(" ")
            #print(str(listOfWords)+" "+str(len(listOfWords)))
            self.output+=";"+listOfWords[3].strip()
            self.upStage()
        #check if we need to restart
        if(self.lastStage()):
                print(self.output)
                self.f2.write(self.output+"\n")
                self.restart()

def main(argv):
    if int(argv[3])==0:
        rP=TLresultFileProcesser(argv[1],argv[2])
        rP.run()
    elif int(argv[3])==1:
        rP=segmenterResultFileProcesser(argv[1],argv[2])
        rP.run()
    elif int(argv[3])==2:
        rP=UnetResultFileProcesser(argv[1],argv[2])
        rP.run()
    elif int(argv[3])==3:
        rP=marianoResultFileProcesser(argv[1],argv[2],[argv[i] for i in range(4,len(argv))] )
        rP.run()
    else:
        raise ValueError("Format exp results, incorrect type of experiment")

if __name__ == '__main__':
    main(sys.argv)
