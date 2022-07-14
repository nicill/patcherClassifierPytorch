import numpy as np
import sys
import dice
import cv2


def main(argv):

    gtMask=cv2.imread(argv[1],cv2.IMREAD_GRAYSCALE)
    if gtMask is None: raise Exception("No ground Truth mask read")

    coarseMask=cv2.imread(argv[2],0)
    if coarseMask is None: raise Exception("No coarse mask read")

    fineMask=cv2.imread(argv[3],0)
    if fineMask is None: raise Exception("No fine mask read")

    for name,mask in [("coarse",coarseMask),("fine",fineMask)]:

        currentDice=dice.dice(mask,255-gtMask )
        currentCoverPerc=dice.coveredPerc(255-gtMask,mask)
        currentFPPerc=dice.FPPerc(255-gtMask,mask)
        print("******************************************* "+name+" mask,  dice: "+str(currentDice)+" and covered Percentage "+str(currentCoverPerc)+" and FP perc "+str(currentFPPerc))




if __name__ == '__main__':
    main(sys.argv)
