import cv2
import numpy as np
from collections import namedtuple
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import os
import sys

from fastai.vision import open_image

mosaicInfo = namedtuple("mosaicInfo","path mosaicFile numClasses layerNameList layerFileList outputFolder " )

def sliding_windowMosaicMask(image,mask, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]],mask[y:y + windowSize[1], x:x + windowSize[0]])

#refine a mask created with patch predictions
def refineMask(mosaic,mask,windowSize,patchSize,mode=0,learner=None,label=""):
    step=windowSize#change if we want to overlap windwos
    #SHOULD WE MAKE THE STEP CHANGE FOR OVERLAPPING PATCHES???

    #retMask=0*np.ones((mosaic.shape[0],mosaic.shape[1],1),dtype=np.uint8)
    nonWhiteTH=0.9
    nonMaskedTH=0.05
    totalPixelsWindow=windowSize*windowSize
    workMask = np.zeros(mosaic.shape[:2], dtype = "uint8")
    count=0#debugging purposes!!!!

    for (x, y, mosaicW,maskW) in sliding_windowMosaicMask(mosaic,mask, stepSize=step, windowSize=(windowSize, windowSize)):
        #only refine windows with enough non-white pixels that also have enough mask pixels
        okPixelsMosaic=np.sum(mosaicW != (255,255,255))
        masked=np.sum(maskW != 0)
        if (okPixelsMosaic>nonWhiteTH*totalPixelsWindow) and (masked>nonMaskedTH*totalPixelsWindow):
            # modes: 0, refine with superpixels, 1 refine with disjoint patches, 2 refine with overlaping patches (heatmap)
            if mode==0: refineMaskWithSuperpixels(mosaicW,maskW,patchSize,count) #not returning anything, should already be changing the correct data
            elif mode==1: refineMaskWithSmallerPatches(mosaicW,maskW,patchSize,learner,count,label) #not returning anything, should already be changing the correct data
            elif mode==2:
                workMaskW=workMask[y:y + windowSize, x:x + windowSize]
                refineMaskWithSmallerPatchesOverlapping(mosaicW,maskW,workMaskW,patchSize,learner,count,label) #not returning anything, should already be changing the correct data
            else: raise Exception("imagePatcherAnnotator refineMask, mask refinement mode not found:"+str(mode))
        count+=1

    if mode==2:
        wMaskMax=np.max(workMask)
        heatMapPerc=0.5
        workMask[workMask<int(wMaskMax*heatMapPerc)]=0
        workMask[workMask!=0]=255
        print("workmask max!! "+str(wMaskMax))
        return workMask

    io.imsave("./shit/FINALMASK.jpg", mask)

def refineMaskWithSuperpixels(mosaicW,maskW,patchSize,j):
    # receive a coarse mask, compute smallish superpixels,
    # only keep as painted those that are almost totally covered by the mask

    coverThresholdPC=0.9
    sPside=20
    inMaskTH=0.9
    pixelsPerSP=sPside*sPside
    #HOW many pixels fit in my mask?
    ### SLIC SEGMENTATION
    numSegments = mosaicW.shape[0]*mosaicW.shape[1]//pixelsPerSP
    vectCompact = 20 #` 15 # 8
    maxIter = 50
    sigMa = 1

    imageSP = img_as_float(mosaicW)
    #print(imageSP.shape)

    #print("starting superpixles")
    segments = slic(imageSP, n_segments=numSegments, compactness=vectCompact, convert2lab=True, sigma=sigMa)
    #print("finishing superpixles")
    #io.imsave("./shit/papap"+str(j)+".jpg", mark_boundaries(imageSP, segments))
    #io.imsave("./shit/papap"+str(j)+"MASK.jpg", maskW)

    #now, traverse superpixels and keep those that are sufficiently inside of the mask.
    for (i, segVal) in enumerate(np.unique(segments)):
        # construct a mask for the segment
        pixelMask = np.zeros(mosaicW.shape[:2], dtype = "uint8")
        #count how many pixels are there in the segment
        pixelMask[segments == segVal] = 255
        totalSPPixels=np.sum(pixelMask == 255)
        # now paint black those that are not in the mask
        pixelMask[maskW==0]=0
        inMaskSPPixels=np.sum(pixelMask == 255)
        inMaskRatio=inMaskSPPixels/totalSPPixels
        #print(" inratio: "+str(inMaskRatio)+" "+str(inMaskSPPixels)+" "+str(totalSPPixels))
        #if not enough pixels of the superpixel are covered, erase this part of the mask
        if not inMaskRatio > inMaskTH:maskW[segments == segVal]=0

def refineMaskWithSmallerPatches(mosaicW,maskW,patchSize,learner,count,label):
    if learner==None: raise Exception("imagePatcherAnnotator, refineMaskWithSmallerPartches, no learner! ")
    if patchSize>50:innerPatchSize=50
    else: innerPatchSize=patchSize//2

    step=innerPatchSize
    pixelsInnerPatch=innerPatchSize*innerPatchSize
    innerMaskedTH=0.1
    # Take small patches inside of this window, possibly overlapping
    for (x, y, innerMosaicPatch,innerMaskPatch) in sliding_windowMosaicMask(mosaicW,maskW, stepSize=step, windowSize=(innerPatchSize, innerPatchSize)):

        # Check that this is a patch that is marked as belonging to the class
        masked=np.sum(innerMaskPatch != 0)
        keep=False
        if masked>innerMaskedTH*pixelsInnerPatch:
            #this is a patch that has been predicted to belong to the class
            #reshape and classify it,
            newPatch = cv2.resize(innerMosaicPatch,(patchSize,patchSize))
            cv2.imwrite("./tempImage.jpg",newPatch)
            img=open_image("./tempImage.jpg")

            pred_class,pred_idx,outputs =learner.predict(img)
            #if it belong to the class, keep it
            if label in str(pred_class).split(";"):keep=True

        if not keep:  innerMaskPatch.fill(0) #If we do are not keeping the patch, paint it black.

def refineMaskWithSmallerPatchesOverlapping(mosaicW,maskW,workMaskW,patchSize,learner,count,label):
    if learner==None: raise Exception("imagePatcherAnnotator, refineMaskWithSmallerPartches, no learner! ")
    if patchSize>50:innerPatchSize=50
    else: innerPatchSize=patchSize//2

    step=10
    pixelsInnerPatch=innerPatchSize*innerPatchSize
    innerMaskedTH=0.1
    # Take small patches inside of this window, possibly overlapping
    for (x, y, innerMosaicPatch,innerMaskPatch) in sliding_windowMosaicMask(mosaicW,maskW, stepSize=step, windowSize=(innerPatchSize, innerPatchSize)):

        # create a mask of ones to be added each time a patch is found to contain the label
        maskOfOnes = 1*np.ones((innerMaskPatch.shape[0],innerMaskPatch.shape[1]), dtype = "uint8")

        # Check that this is a patch that is marked as belonging to the class
        masked=np.sum(innerMaskPatch != 0)
        keep=False
        if masked>innerMaskedTH*pixelsInnerPatch:
            #this is a patch that has been predicted to belong to the class
            #reshape and classify it,
            newPatch = cv2.resize(innerMosaicPatch,(patchSize,patchSize))
            cv2.imwrite("./tempImage.jpg",newPatch)
            img=open_image("./tempImage.jpg")

            pred_class,pred_idx,outputs =learner.predict(img)
            #if it belong to the class, keep it
            if label in str(pred_class).split(";"):keep=True

        if keep:  #If we do are not keeping the patch, paint it black.
            innerWorkMaskW=workMaskW[y:y + innerPatchSize, x:x + innerPatchSize]
            workMaskW[y:y + innerPatchSize, x:x + innerPatchSize]=np.add(innerWorkMaskW,maskOfOnes)

def createLayerMask(predictionFile,patchSize,imShape,category):

    #print("createLayerMask "+str(predictionFile)+" "+str(patchSize)+" "+str(imShape)+" "+str(category))

    shapeX=imShape[0]
    shapeY=imShape[1]

    numStepsX=int(shapeX/patchSize)
    numStepsY=int(shapeY/patchSize)

    # create empty mask
    retVal=np.zeros((shapeX,shapeY),dtype=np.uint8)

    f=open(predictionFile,"r")
    for line in f:
        # get current image name  image
        imName=line.split(",")[0]
        if imName!="image":
            # get patch number and prediction list
            patchNumber=int(line.split("h")[1].split(" ")[0])
            if len(line.strip().split(" "))>1:
                predictionList=line.strip().split(" ")[1].split(";")
            else:
                predictionList=[]
            #print("createLayerMask patch and list "+str(patchNumber)+" "+str(predictionList))

            #now, if the patch contains the category we are looking for, paint the corresponding patch black
            if category in predictionList:
                #print("createLayerMask found category in patch and list "+str(patchNumber)+" "+str(predictionList))
                xJump=patchNumber//numStepsY
                yJump=patchNumber%numStepsY
                paintImagePatch(retVal,xJump*patchSize,yJump*patchSize,patchSize,255)

    return retVal

def imagePatch(image,minX,minY,size,verbose=False):
    if(verbose):print("making patch [ "+str(minX)+" , "+str(minX+size)+"] x [ "+str(minY)+" , "+str(minY+size)+"]")

    return image[minX:minX+size, minY:minY+size]

def paintImagePatch(image,minX,minY,size,color,verbose=False):
    if(verbose):print("painting patch [ "+str(minX)+" , "+str(minX+size)+"] x [ "+str(minY)+" , "+str(minY+size)+"]")

    image[minX:minX+size, minY:minY+size]=color


def imageNotEmpty(image):
    pixelThreshold=500
    #whitePixels=np.sum(image==255)
    blackPixels=np.sum(image==0) #black pixels contains class information
    #totalPixels=(image.shape[0]*image.shape[1])
    #nonwhitePixels=totalPixels-whitePixels
#    return nonwhitePixels>whitePixelThreshold
    return blackPixels>pixelThreshold

def notWhite(pixel):
    return pixel!=255
    #return (pixel[0]!=255 or pixel[1]!=255 or pixel[2]!=255)

def listPixels(image):
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[0]):
            print ("pixel "+str(image[i][j]))
            #if(notWhite(image[i][j])):return True
    return False

def interpretParameters(paramFile,verbose=False):
    # read the parameter file line by line
    f = open(paramFile, "r")
    patchSize=-1
    mosaicDict={}
    train=[]
    important=[]
    unImportant=[]
    testFileN=None

    for x in f:
        lineList=x.split(" ")
        # read every line
        first=lineList[0]

        if first[0]=="#": #if the first character is # treat as a comment
            if verbose:print("COMMENT: "+str(lineList))
        elif first=="\n":# account for blank lines, do nothing
            pass
        elif first=="patchSize":
            patchSize=int(lineList[1].strip())
            if verbose:print("Read Patch Size : "+str(patchSize))
        elif first=="csvFileName":
            csvFileN=lineList[1].strip()
            if verbose:print("Read csv file name : "+csvFileN)
        elif first=="testFileName":
            testFileN=lineList[1].strip()
            if verbose:print("Read test file name : "+csvFileN)
        elif first=="mosaic":
            layerNameList=[]
            layerList=[]

            # read the number of layers and set up reading loop
            filePath=lineList[1]
            mosaic=lineList[2]
            numClasses=int(lineList[3])
            outputFolder=lineList[4+numClasses*2].strip()
            for i in range(4,numClasses*2+3,2):
                layerNameList.append(lineList[i])
                layerList.append(filePath+lineList[i+1])

            #make dictionary entry for this mosaic
            mosaicDict[mosaic]=mosaicInfo(filePath,mosaic,numClasses,layerNameList,layerList,outputFolder)
            if verbose:
                print("\n\n\n")
                print(mosaicDict[mosaic])
                print("\n\n\n")
                #print("Read layers and file : ")
                #print("filePath "+filePath)
                #print("mosaic "+mosaic)
                #print("num Classes "+str(numClasses))
                #print("layerName List "+str(layerNameList))
                #print("layer List "+str(layerList))
                #print("outputFolder "+outputFolder)
        elif first=="train":
            for x in lineList[1:]:
                if x.strip() not in train:
                    train.append(x.strip())
        elif first=="important":
            for x in lineList[1:]:
                if x.strip() not in important:
                    important.append(x.strip())
        elif first=="unimportant":
            for x in lineList[1:]:
                if x.strip() not in unImportant:
                    unImportant.append(x.strip())
        else:
            raise Exception("ImagePatchAnnotator:interpretParameters, reading parameters, received wrong parameter "+str(lineList))

        if verbose:(print(mosaicDict))

    return patchSize,csvFileN,testFileN,mosaicDict,train,important,unImportant

def readImage(imageName,mode,verbose=False):
    if verbose: print("readImage::Reading "+imageName)
    if mode=="color":image=cv2.imread(imageName,cv2.IMREAD_COLOR)
    elif mode=="grayscale":image=cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
    else: raise Exception("imagePatcherAnnotator:readImage, wrong image mode")

    if image is None: raise Exception("ImagePatchAnnotator:readImage image not found "+imageName)
    return image

def mosaicToPatches(mInfo, patchsize, csvFileName,firstMosaic=False,verbose=False):
    #mosaicInfo = namedtuple("mosaicInfo","path mosaicFile numClasses layerNameList layerFileList outputFolder " )

    layerNames=mInfo.layerNameList
    imageName=mInfo.path+"/"+mInfo.mosaicFile

    f = open(csvFileName, "a")
    image = readImage(imageName,"color",verbose)

    shapeX=image.shape[0]
    shapeY=image.shape[1]

    outputDir=mInfo.path+mInfo.outputFolder+"/"
    if firstMosaic:
        try:
            # Create target Directory
            os.mkdir(outputDir)
            print("Directory " , outputDir ,  " Created ")
        except FileExistsError:
            print("Directory " , outputDir ,  " already exists, this should probably not be happening, erase it first unless you have a good reason not to!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    outputPrefix=mInfo.mosaicFile[:-4]

    if verbose:print("OUTPUTDIR "+outputDir)
    if verbose:print("OUTPUTPrefix "+outputPrefix)

    #print("image shape "+str(shapeX)+","+str(shapeY))

    #first, make the patches fully inside the image
    numStepsX=int(shapeX/patchsize)
    numStepsY=int(shapeY/patchsize)

    if verbose: print("steps "+str(numStepsX)+" "+str(numStepsY))
    count=0
    if firstMosaic: f.write("image,tags")

    #read all layer images
    layerList=[]
    for x in range(0,len(layerNames)):
        # check if the layer in this patch is empty
        layerName=mInfo.layerFileList[x]
        if verbose: print("Reading "+layerName)
        #layerList.append(cv2.imread(layerName,cv2.IMREAD_GRAYSCALE))
        layerList.append(readImage(layerName,"grayscale",verbose))

    for i in range(0,numStepsX):
        if verbose:print("i is now "+str(i)+" of "+str(numStepsX))
        for j in range(0,numStepsY):
            if verbose:print("             j is now "+str(j)+" of "+str(numStepsY))
            # create patch image and write it

            # also check how many layers are non-empty in this patch
            layerString=""
            for x in range(0,len(layerList)):
                # check if the layer in this patch is empty
                if layerList[x] is None :break

                layer=layerList[x]
                layerPatch=imagePatch(layer ,i*patchsize,j*patchsize,patchsize,False  )

                #cv2.imwrite("./wm1/patches/patch"+str(count)+"Layer"+str(x)+".jpg",layerPatch)
                if( imageNotEmpty( layerPatch )):
                    layerString+=layerNames[x]+" "
                    #print("updating layer string for "+layerString)

            if(not layerString=="") :
                outputImageName=outputDir+"/"+outputPrefix+"patch"+str(count)+".jpg"
                #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GOING TO WRITE IMAGE TO FILE "+outputImageName)
                cv2.imwrite(outputImageName, imagePatch(image,i*patchsize,j*patchsize,patchsize) )
                if verbose: print("for patch "+str(count)+" string is "+layerString)
                f.write("\n"+outputPrefix+"patch"+str(count)+","+layerString.strip())
                f.flush()
            count+=1

    f.close()

def mosaicToPixelWisePatches(mInfo, patchsize, csvFileName,training=True,firstMosaic=False,verbose=False):
    #mosaicInfo = namedtuple("mosaicInfo","path mosaicFile numClasses layerNameList layerFileList outputFolder " )

    layerNames=mInfo.layerNameList
    imageName=mInfo.path+"/"+mInfo.mosaicFile

    image = readImage(imageName,"color",verbose)

    shapeX=image.shape[0]
    shapeY=image.shape[1]

    f = open(csvFileName, "a")

    if training:outputDir=mInfo.path+mInfo.outputFolder+"/"
    else:outputDir=mInfo.path+"TEST"+mInfo.outputFolder+"/"

    if firstMosaic:
        f.write("image,tags")
        try:
            # Create target Directory
            os.mkdir(outputDir)
            print("Directory " , outputDir ,  " Created ")
            os.mkdir(outputDir+'/labels')
            os.mkdir(outputDir+'/images')

        except FileExistsError:
            print("Directory " , outputDir ,  " already exists, this should probably not be happening, erase it first unless you have a good reason not to!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            try:
                os.mkdir(outputDir+'/labels')
            except Exception as e:
                print(" labels file already existed!!!! "+str(e))
            try:
                os.mkdir(outputDir+'/images')
            except Exception as e:
                print(" images file already existed!!!! "+str(e))

    outputPrefix=mInfo.mosaicFile[:-4]

    if verbose:print("OUTPUTDIR "+outputDir)
    if verbose:print("OUTPUTPrefix "+outputPrefix)

    #print("image shape "+str(shapeX)+","+str(shapeY))

    #first, make the patches fully inside the image
    numStepsX=int(shapeX/patchsize)
    numStepsY=int(shapeY/patchsize)

    if verbose: print("steps "+str(numStepsX)+" "+str(numStepsY))
    count=0

    #read all layer images and initialize colors
    layerColors=[]

    # do not read the last layer as it is "void"
    layerList=[]
    for x in range(0,len(layerNames)-1):
        # check if the layer in this patch is empty
        layerName=mInfo.layerFileList[x]
        if verbose: print("Reading "+layerName)
        layerList.append(readImage(layerName,"grayscale",verbose))
        layerColors.append(x+1)
    #  add also color "0" for Void layer and add a "None image" to the list of layers
    layerColors.append(0)
    layerList.append(None)

    print("layer Names "+str(mInfo.layerFileList))
    print("layer Colors "+str(layerColors))

    for i in range(0,numStepsX):
        if verbose:print("i is now "+str(i)+" of "+str(numStepsX))
        for j in range(0,numStepsY):
            if verbose:print("             j is now "+str(j)+" of "+str(numStepsY))
            # create patch image and write it

            # also check how many layers are non-empty in this patch
            layerString=""
            #create empty patch image
            segmPatch=np.zeros((patchsize,patchsize),dtype=np.uint8)
            segmPatch.fill(0)

            for x in range(0,len(layerList)):
                # check if the layer in this patch is empty
                if layerList[x] is None :break

                layer=layerList[x]
                layerPatch=imagePatch(layer ,i*patchsize,j*patchsize,patchsize,False  )

                #cv2.imwrite("./wm1/patches/patch"+str(count)+"Layer"+str(x)+".jpg",layerPatch)
                if( imageNotEmpty( layerPatch )):
                    layerString+=layerNames[x]+" "
                    #print("updating layer string for "+layerString)
                    #as this layer is not empty, color the pertinent part of the segmenation patch
                    segmPatch[layerPatch==0]=layerColors[x]

            if(not layerString=="") :
                outputImageName=outputDir+"/labels/"+outputPrefix+"SegmentationPatch"+str(count)+".png"
                if (segmPatch<0).sum()>0 or (segmPatch>len(layerList)).sum()>0:
                    print("WTF")
                    sys.exit(-1)
                cv2.imwrite(outputImageName, segmPatch )
                #to write the real patch uncomment the following two lines
                outputImageName2=outputDir+"/images/"+outputPrefix+"SegmentationPatch"+str(count)+"Real.png"
                cv2.imwrite(outputImageName2, imagePatch(image,i*patchsize,j*patchsize,patchsize) )
                if verbose: print("for patch "+str(count)+" string is "+layerString)
                f.write("\n"+outputPrefix+"SegmentationPatch"+str(count)+"Real.png"+","+layerString.strip())
                f.flush()
            count+=1

    f.close()




def main(argv):

    verbose=False
    unetPatches=True
    patchSize,csvFileName,testFileName,mosaicDict,train,important=interpretParameters(argv[1])

    #if verbose: print(mosaicDict)
    firstMosaic=True
    for k,v in mosaicDict.items():
        if verbose: print("\n\nstarting processing of first mosaic and layers "+str(v)+"\n\n")
        if unetPatches:
            v.layerNameList.append("Void")
            mosaicToPixelWisePatches(v, patchSize, csvFileName, firstMosaic,verbose)
        else:
            mosaicToPatches(v, patchSize, csvFileName,firstMosaic,verbose)


        firstMosaic=False


if __name__== "__main__":
  main(sys.argv)
