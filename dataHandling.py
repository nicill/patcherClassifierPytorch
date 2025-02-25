import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2

import numpy as np
import pandas as pd
import sys
from pathlib import Path


# re library to extract information quickly from file names
import re



# Let's define functions to read RGB and binary images using OpenCv
def readMosaic(fileName,convert = True):
    """
        Function to read RGB images and transform them
        from RBG (opencv) to RGB (matplotlib)
    """
    retVal = cv2.imread(fileName,cv2.IMREAD_COLOR)
    # As opencv stores images BGR instead of RGB,
    # we need to change them back to visualize them
    return retVal[:,:,::-1] if convert else retVal

def readBM(fileName):
    """
        Function to read a binary image from a file
    """
    retVal = cv2.imread(fileName,0)
    return retVal

def create_Patch(mosaic, center ,patchSize,convert = False):
    """
        Function that receives:
            a large color image (mosaic)
            the coordinates of a point (center)
            a patch size
        And returns a square patch of the indicated size
        around the coordinates of the center
        (convert idicates whether it is necessary to invert the pixel colors)
    """
    def get_Square(w_size, p, img):
        """
            Internal function to cut a square patch
        """
        retVal = img[int(p[0])-w_size//2:int(p[0])+w_size//2, int(p[1])-w_size//2:int(p[1])+w_size//2]
        if len(retVal) == 0: raise Exception("NONE is "+str(p)+" "+str(w_size))
        return retVal
    patchimage = get_Square(patchSize, (center[1],center[0]), mosaic)
    return cv2.cvtColor(patchimage, cv2.COLOR_BGR2RGB) if convert else patchimage


def patchesFromMosaicAndCoords(infoFile, mosaicFile, outFolder, size = 50):
    """
        Receives a csv with information on coordinates and 
        weed species and a mosaic 
        creates one patch for every weed in the proper coordinates
    """
    def processAPatch(tup):
        """
            Internal image to store every patch created
            with properName
        """
        nonlocal patchNum
        cat, x, y = tup
        im = create_Patch(mosaic, (y,x) ,size) #carefull with x,y y,x!!!!!!!!!!!!!!!!!!!!!!!
        cv2.imwrite(os.path.join(outFolder,"patch"+str(patchNum)+"sp"+str(cat)+".png"), im)
        patchNum+=1

    # create output directory if it does not exist
    Path(outFolder).mkdir(parents=True, exist_ok=True)

    # read the data
    data = pd.read_csv(infoFile)
    print("Reading mosaic")
    mosaic = readMosaic(mosaicFile, convert = False)
    print("mosaic Read")

    patchNum = 0
    # create tuyples of patches and categories and store
    #catsAndPatches = [ (cat, create_Patch(mosaic, (x,y) ,size)  )      for cat, x, y in zip(data['Code_weed'], data['px'],data['py']) ]

    # 
    list(map(processAPatch , zip(data['Code_weed'], data['px'],data['py']) ))


if __name__ == '__main__':
    patchesFromMosaicAndCoords(sys.argv[1],sys.argv[2],sys.argv[3],100)
