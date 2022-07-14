import matplotlib.pyplot as plt
import csv
import sys

def main(argv):

    x = []
    y = []
    deleteList=["LR","ALEXNET", "SQUEEZENET", "VGG"]
#ResNet50
#ResNet152
#VGG
#densenet
#ResNext

    #print("opening "+str(argv[1]))
    yLabel=""
    xLabel=[]
    plt.rcParams.update({'font.size': 28})
    plt.ylim(top=96, bottom=70)
    #plt.figure(dpi=800)
    with open(argv[1],'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            print(row)

            #first row, find out number of categories
            if yLabel=="":
                yLabel=row[0]
                for i in range(1,len(row)):
                    x.append([])
                    xLabel.append(row[i])
                print(xLabel)
            else:
                #   y.append(float(row[0]))
                for i in range(len(row)-1):
                    x[i].append(float(row[i+1]))
    #sys.exit()
    print(x)
    dataList=[]
    labelList=[]
    for i in range(len(x)):
        #print("\n\n plotting "+str(x[i]))
        #plt.plot(y,x[i], label=xLabel[i])
        if xLabel[i] not in deleteList:
            print(xLabel[i])
            labelList.append(xLabel[i])
            dataList.append(x[i])   
    print(labelList)
    plt.boxplot(dataList, showmeans=True, notch=True, labels=labelList,  showfliers=True)
    #plt.xticks( rotation='vertical')
    # see, for reference https://matplotlib.org/3.1.0/gallery/statistics/boxplot.html
    code=int(argv[2])

    labelsLong=["True Positive Rate","False Positive Rate","Accuracy"]
    labelsShort=["TPR","FPR","ACC"]

    #plt.xlabel("Networks")
    #plt.ylabel(labelsShort[code]+" %")
   # plt.xscale('log')
    #plt.title(labelsLong[code]+' %')
    #plt.legend(prop={'size': 0})
    plt.legend(loc=0)
    plt.show()

    #https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
    #‘best’ 	0 ‘upper right’ 	1 ‘upper left’ 	2 ‘lower left’ 	3 ‘lower right’ 	4 ‘right’ 	5
    #‘center left’ 	6 ‘center right’ 	7 ‘lower center’ 	8 ‘upper center’ 	9 ‘center’ 	10

    #plt.savefig(argv[3])

if __name__ == '__main__':
    main(sys.argv)
