import argparse
import sys
import os
import re
import cv2
import time
import random
import numpy as np
from torch.utils.data import DataLoader
from data_manipulation.utils import color_codes, find_file
from datasets import Cropping2DDataset
from models import  myFeatureExtractor

import torch
import torchvision
from torchvision import models as torchModels
import torch.nn as nn

import dice
import imagePatcherAnnotator as pa

def setRandomSeed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
#Remember to use num_workers=0 when creating the DataBunch.

def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')

    # Mode selector
    parser.add_argument(
        '-d', '--mosaics-directory',
        dest='val_dir', default='/media/yago/workDrive/Experiments/forests/floorPatcherPytorch/crown',
        help='Directory containing the mosaics'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '-arch', '--architecture',
        dest='arch',
        default="res",
        help='type of architecture'
    )
    parser.add_argument(
        '-frozen', '--frozen',
        dest='frozen',
        default="True",
        help='whether or no the feature extractor is frozen'
    )
    parser.add_argument(
        '-paramF', '--paramF',
        dest='paramF',
        help='parameter File that will contain the mosaics to read along with their ground truth'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=3,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '-lr', '--learningRate',
        dest='lr',
        type=float, default=0.001,
        help='Learning rate for the optimizer'
    )
    parser.add_argument(
        '-numC', '--NumChannels',
        dest='nChan',
        type=int, default=3,
        help='Number of Channels, 3=RBG, 4 include also DEM'
    )
    parser.add_argument(
        '-B', '--batch-size',
        dest='batch_size',
        type=int, default=32,
        help='Number of samples per batch'
    )
    parser.add_argument(
        '-t', '--patch-size',
        dest='patch_size',
        type=int, default=128,
        help='Patch size'
    )
    parser.add_argument(
        '-l', '--labels-tag',
        dest='lab_tag', default='tree',
        help='Tag to be found on all the ground truth filenames'
    )

    options = vars(parser.parse_args())

    return options

def imagePatch(image,minX,minY,size,verbose=False):
    if(verbose):print("making patch [ "+str(minX)+" , "+str(minX+size)+"] x [ "+str(minY)+" , "+str(minY+size)+"]")

    return image[minX:minX+size, minY:minY+size]

def binarizeMask(probabilityMask,th=0.2):
    # receives a probability image and turns it into a binary mask
    # the max in the probability mask is computed
    # points over the threshold proportion of this mask are painted white, other are painted black
    max=np.max(probabilityMask)
    probabilityMask[probabilityMask>th*max]=255
    probabilityMask[probabilityMask<255]=0


def countPatchStatsCategory(gtMask,predicted,patchSize):
    shapeX=gtMask.shape[0]
    shapeY=gtMask.shape[0]

    numStepsX=int(shapeX/patchSize)
    numStepsY=int(shapeY/patchSize)

    pixelThreshold=5*(patchSize*patchSize)/100

    #print("pixel Threshold "+str(pixelThreshold))

    totalPatches=0
    TP=0
    FP=0
    TN=0
    FN=0

    #cv2.imwrite("./mask.jpg",gtMask)
    #cv2.imwrite("./predict.jpg",predicted)

    for i in range(numStepsX):
        for j in range(numStepsY):
            patchGT=imagePatch(gtMask, i*patchSize,j*patchSize,patchSize)
            patchPred=imagePatch(predicted ,i*patchSize,j*patchSize,patchSize)
            #cv2.imwrite("./maskpatch"+str(i)+str(j)+".jpg",patchGT)
            #cv2.imwrite("./predictpatch"+str(i)+str(j)+".jpg",patchPred)

            pos=(np.sum(patchGT<50)>pixelThreshold)
            predPos=(np.sum(patchPred>200)>pixelThreshold)#predictions are white but annotations are black
            # eliminate clearly wrong patches, add the file with all the labels and eliminate patches with only 0 in the ground truth
            totalPatches+=1
            if pos and predPos:TP+=1
            elif not pos and predPos:FP+=1
            elif not pos and not predPos:TN+=1
            elif pos and not predPos:FN+=1
            else: raise Exception("something wrong with patch prediction evaluation ")
    return totalPatches,TP,FP,TN,FN

def buildGroundTruthImage(mosaicDict,layerNames,tag):
    imageNameList=[]
    gtLayersDict={}
    for k,v in mosaicDict.items():
        #load mosaic using path and image name
        mosaic=cv2.imread(v.path+"/"+k)
        if mosaic is None: print("not read")
        #Make zeros image with the same size as the mosaic
        labelImage=np.zeros(mosaic.shape[:2], dtype = "uint8")
        #print(str(k)+" \n "+str(v)+"\n")
        #for every layer,
            #load layer file
            #add the code of the layer to the label image
        counter=0
        gtLayerList=[]
        for layerFile,layerName in zip(v.layerFileList,v.layerNameList):
            #print(str(layerFile))
            while layerName!=layerNames[counter]:
                print("skipping layer "+layerNames[counter]+" "+layerName)
                counter+=1
            currLayer=cv2.imread(layerFile,cv2.IMREAD_GRAYSCALE)
            gtLayerList.append(currLayer)
            counter+=1
            labelImage[currLayer==0]=counter#*(255/6)

        #save label image
        imageNameList.append(k[:-4]+tag+".png")
        cv2.imwrite(v.path+"/"+imageNameList[-1],labelImage)
        #also, store current layers in dictionary
        gtLayersDict[k]=gtLayerList

    return np.max(labelImage),imageNameList,gtLayersDict

def train_test_net(init_net_name, verbose=1):
    """

    :param net_name:
    :return:
    """
    # Init
    c = color_codes()
    options = parse_inputs()
    randomSeed=42

    save_model=True
    addScheduler=True
    augment=True
    augmentFactor=12

    # Data loading (or preparation)
    #First, read parameter file for the mosaics, and create semantic segmentation images
    paramFile=options['paramF']
    patchSize,csvFileName,testFileName,mosaicDict,trainUNUSED,important,unimportant=pa.interpretParameters(paramFile)

    mInfo=mosaicDict[list(mosaicDict.keys())[0]]
    layerNames=mInfo.layerNameList# the list of classes is the one in the first mosaic

    codedImportant=[]
    codedUnImportant=[]
    for num in range(len(layerNames)):
        if layerNames[num] in important:
            codedImportant.append(num)
        elif layerNames[num] in unimportant:
            codedUnImportant.append(num)


    print(str(important))
    print(layerNames)
    print(codedImportant)
    print(codedUnImportant)

    numClasses,gt_names,gtDict = buildGroundTruthImage(mosaicDict,layerNames,options['lab_tag'])

    print("number of classes "+str(numClasses))

    # load label File
    d_path = options['val_dir']
    #gt_names = sorted(list(filter(
    #    lambda x: not os.path.isdir(x) and re.search(options['lab_tag'], x),
    #    os.listdir(d_path)
    #)))
    n_folds = len(gt_names)
    cases = [re.search(r'(\d+)', r).group() for r in gt_names]

    #Read number of Channels
    numChannels=options["nChan"]
    print("Number of channels is "+str(numChannels))

    print("gt names "+str(gt_names))
    print("cases "+str(cases))

    print(
            '%s[%s]%s Loading all mosaics and DEMs%s' %
            (c['c'], time.strftime("%H:%M:%S"), c['g'], c['nc'])
    )
    gtImages = [cv2.imread(os.path.join(d_path, im),0).astype(np.uint8) for im in gt_names]

    #label1=np.sum(gtImages[0]>3)

    #y = [
    #    (np.mean(cv2.imread(os.path.join(d_path, im)), axis=-1) < 2).astype(np.uint8)
    #    for im in gt_names
    #]

    y=gtImages

    if numChannels==4:
        dems = [
            #cv2.imread(os.path.join(d_path, 'DEM{:}.jpg'.format(c_i)))
            cv2.imread(os.path.join(d_path, 'b{:}DEM.jpg'.format(c_i)))
            for c_i in cases
        ]
    else: dems=[]
    print("number of read DEM "+str(len(dems)))

    mosaics = [
        #cv2.imread(os.path.join(d_path, 'mosaic{:}.jpg'.format(c_i)))
        cv2.imread(os.path.join(d_path, 'b{:}.jpg'.format(c_i)))
        for c_i in cases
    ]
    mosaic_names=['b{:}.jpg'.format(c_i) for c_i in cases]
    print("number of read mosaics "+str(len(mosaics)))
    print(" mosaic names "+str(mosaic_names))
    if numChannels==4:
        x = [
            np.moveaxis(
                np.concatenate([mosaic, np.expand_dims(dem[..., 0], -1)], -1),
                -1, 0
            )
            for mosaic, dem in zip(mosaics, dems)
        ]
    else:#numChannels==3
        x = [np.moveaxis(mosaic,-1, 0) for mosaic in mosaics]

    print("number of x "+str(len(x)))

    print(
        '%s[%s] %sStarting cross-validation (leave-one-mosaic-out)'
        ' - %d mosaics%s' % (
            c['c'], time.strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )
    #start large test
    #lrList=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009]
    lrList=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009]

    for learningRate in lrList:
        resultsList=[]
        for i, case in enumerate(cases):
            print(
                '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%s[%s]%s Starting process for mosaic %s %s(%d/%d)%s' %
                (
                    c['c'], time.strftime("%H:%M:%S"),
                    c['g'], case,
                    c['c'], i + 1, len(cases), c['nc']
                )
            )

            if verbose > 0:
                print(
                    '%s[%s]%s Starting training for mosaic %s %s(%d/%d)%s' %
                    (
                        c['c'], time.strftime("%H:%M:%S"),
                        c['g'], case,
                        c['c'], i + 1, len(cases), c['nc']
                    )
                )

            test_y = y[i]
            test_x = x[i]

            train_y = y[:i] + y[i + 1:]
            train_x = x[:i] + x[i + 1:]

            #val_split = 0.1
            val_split = 0
            batch_size = int(options["batch_size"])
            patch_size = (256, 256)
            if options["patch_size"] is not None: patch_size = (int(options["patch_size"]), int(options["patch_size"]))
            sideOfPatch=options["patch_size"]
            overlap = (0, 0)
            num_workers = 1

            print("starting "+str(patch_size))
            architecture=options['arch']
            #learningRate=options['lr']
            if options['frozen']=="True":frozenInit=True
            else : frozenInit=False

            net_name=init_net_name+"patch"+str(sideOfPatch)+"LR"+str(learningRate)+"net"+architecture+"frozen"+str(frozenInit)

            model_name = '{:}.mosaic{:}.mdl'.format(net_name, case)

            print("initializing feature extractor with architecture "+str(architecture)+" learningRate "+str(learningRate)+" and frozen "+str(frozenInit))

            net = myFeatureExtractor(n_outputs=numClasses,frozen=frozenInit,featEx=architecture,LR=learningRate,nChanInit=numChannels)

            training_start = time.time()
            try:
                net.load_model(os.path.join(d_path, model_name))
            except IOError:

                # Dataloader creation
                if verbose > 0:
                    n_params = sum(
                        p.numel() for p in net.parameters() if p.requires_grad
                    )
                    print(
                        '%sStarting training %s (%d parameters)' %
                        (c['c'], c['nc'], n_params)
                    )

                if val_split > 0:
                    n_samples = len(train_x)

                    n_t_samples = int(n_samples * (1 - val_split))

                    d_train = train_x[:n_t_samples]
                    d_val = train_x[n_t_samples:]

                    l_train = train_y[:n_t_samples]
                    l_val = train_y[n_t_samples:]

                    print('Training dataset (with validation) '+str(val_split)+" "+str(n_samples)+" "+str(n_t_samples))
                    setRandomSeed(42)
                    train_dataset = Cropping2DDataset(
                        d_train, l_train, patch_size=patch_size, overlap=overlap,nChan=numChannels,augmentFactor=augmentFactor,interestingClasses=codedImportant,uninterestingClasses=codedUnImportant,augment=augment
                    )

                    print('Validation dataset (with validation)')
                    setRandomSeed(42)
                    val_dataset = Cropping2DDataset(
                        d_val, l_val, patch_size=patch_size, overlap=overlap
                    )
                else:
                    print('Training dataset')
                    setRandomSeed(42)
                    train_dataset = Cropping2DDataset(
                        train_x, train_y, patch_size=patch_size, overlap=overlap,nChan=numChannels,augmentFactor=augmentFactor,interestingClasses=codedImportant,uninterestingClasses=codedUnImportant,augment=augment
                    )

                    print('Validation dataset')
                    setRandomSeed(42)
                    val_dataset = Cropping2DDataset(
                        train_x, train_y, patch_size=patch_size, overlap=overlap
                    )

                setRandomSeed(42)
                train_dataloader = DataLoader(
                    train_dataset, batch_size, True, num_workers=num_workers
                )
                setRandomSeed(42)
                val_dataloader = DataLoader(
                    val_dataset, batch_size, num_workers=num_workers
                )

                setRandomSeed(42)
                epochs = parse_inputs()['epochs']
                patience = parse_inputs()['patience']

                if addScheduler:
                    net.addScheduler(learningRate,len(train_dataloader),epochs)

                net.fit(
                    train_dataloader,
                    val_dataloader,
                    epochs=epochs,
                    patience=patience,
                    verbose=True
                )

                if save_model: net.save_model(os.path.join(d_path, model_name))

            if verbose > 0:
                time_str = time.strftime(
                    '%H hours %M minutes %S seconds',
                    time.gmtime(time.time() - training_start)
                )
                print(
                    '%sTraining finished%s (total time %s)\n' %
                    (c['r'], c['nc'], time_str)
                )

                print(
                    '%s[%s]%s Starting testing with mosaic %s %s(%d/%d)%s' %
                    (
                        c['c'], time.strftime("%H:%M:%S"),
                        c['g'], case,
                        c['c'], i + 1, len(cases), c['nc']
                    )
                )

            yi = net.test(data=[test_x],patch_size=patch_size[0],verbose=True,refine=True,classesToRefine=[0])

            #change so this also takes numclasses arguments and loops over them
            #also be careful that the dice is computed over the correct mask
            #gtIm=gtImages[i]
            #print("lenght of yi "+str(len(yi)))
            #print(yi)

            for counter in range(numClasses):
                print("starting prediction of "+str(layerNames[counter]))
                #print(str(mosaicDict[mosaic_names[i]]))

                predIm=yi[counter][0]
                currentGtImage=gtDict[mosaic_names[i]][counter]
                #currentGtImage=cv2.imread(mosaicDict[mosaic_names[i]].layerFileList[counter],0)

                if counter>0:
                    #cv2.imwrite(
                    #    os.path.join(d_path, "NONBpatch"+str(sideOfPatch)+"LR"+str(learningRate)+"net"+architecture+"frozen"+str(frozenInit)+"layer"+str(counter)+'pred_{:}.jpg'.format(case)),
                    #    predIm
                    #)
                    cv2.imwrite(
                        "./NONBpatch"+str(sideOfPatch)+"LR"+str(learningRate)+"net"+architecture+"frozen"+str(frozenInit)+"layer"+str(counter)+'pred_{:}.jpg'.format(case),
                        predIm
                    )

                binarizeMask(predIm)
                total,TP,FP,TN,FN=countPatchStatsCategory(currentGtImage,predIm,patch_size[0])
                currentResults=(total,TP,FP)

                print(str(layerNames[counter])+" TPR "+str(100*TP/(TP+FN)))
                print(str(layerNames[counter])+" FPR "+str(100*FP/(TN+FP)))
                print(str(layerNames[counter])+" ACCURACY "+str(100*(TP+TN)/(total)))
                #print(str(layerNames[counter])+" TNR "+str(100*TN/(TN+FP))  )
                #print(str(layerNames[counter])+" FNR "+str(100*FN/(TP+FN)))
                print(str(layerNames[counter])+" Stats: "+str(total)+" "+str(TP)+" "+str(FP)+" "+str(TN)+" "+str(FN))

                #print("Writing "+str("patch"+str(sideOfPatch)+"LR"+str(learningRate)+"net"+architecture+"frozen"+str(frozenInit)+str(counter)+'pred_{:}.jpg'.format(case)))
                if counter>0:
                    cv2.imwrite(
                        os.path.join(d_path, "patch"+str(sideOfPatch)+"LR"+str(learningRate)+"net"+architecture+"frozen"+str(frozenInit)+"layer"+str(counter)+'pred_{:}.jpg'.format(case)),
                        predIm
                    )

                #sys.exit()

                print("mosaic"+str(case)+" DICE COEFFICIENT :"+str(dice.dice(255-currentGtImage,255-predIm)),flush=True)
                #print("mosaic"+str(case)+" DICE COEFFICIENT2 :"+str(dice.dice(currentGtImage,predIm)))
                #print("mosaic"+str(case)+" DICE COEFFICIENT3 :"+str(dice.dice(currentGtImage,255-predIm)))
            resultsList.append(currentResults)

        sumTotal=0
        sumTP=0
        sumFP=0
        for el in resultsList:
            print("starting first tuple "+str(el))
            sumTotal+=el[0]
            sumTP+=el[1]
            sumFP+=el[2]
        print("FINISHED LEAVE ONE OUT FOR LR"+str(learningRate)+" averageTotal "+str(sumTotal/len(resultsList))+" average FP "+str(sumTP/len(resultsList))+" average FP "+str(sumFP/len(resultsList)))


def main():
    # Init
    c = color_codes()

    print(torchvision.__version__)

    print(
        '%s[%s] %s<Tree detection pipeline>%s' % (
            c['c'], time.strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    ''' <Detection task> '''
    net_name = 'floor-detection'

    train_test_net(net_name)


if __name__ == '__main__':
    main()
