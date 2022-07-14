import numpy as np
import random
import cv2

from torch.utils.data.dataset import Dataset
from data_manipulation.datasets import get_slices_bb

from imgaug import augmenters as iaa


def augment(image,code,verbose=True):
    #outputFile="./fuio.jpg"
    #print("shape!!! "+str(image.shape))
    if code==0:
        if verbose: print("Doing Data augmentation 0 (H fip) to image ")
        image_aug = iaa.Rot90(1)(image=image)
        #cv2.imwrite(outputFile,image_aug)
    elif code==1:
        if verbose: print("Doing Data augmentation 1 (V flip) to image ")
        image_aug = iaa.Flipud(1.0)(image=image)
        #cv2.imwrite(outputFile,image_aug)
    elif code==2:
        if verbose: print("Doing Data augmentation 2 (Gaussian Blur) to image ")
        image_aug = iaa.GaussianBlur(sigma=(0, 0.5))(image=image)
        #cv2.imwrite(outputFile,image_aug)
    elif code==3:
        if verbose: print("Doing Data augmentation 3 (rotation) to image ")
        angle=random.randint(0,45)
        rotate = iaa.Affine(rotate=(-angle, angle))
        image_aug = rotate(image=image)
        #cv2.imwrite(outputFile,image_aug)
    elif code==4:
        if verbose: print("Doing Data augmentation 4 (elastic) to image ")
        image_aug = iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.1)(image=image)
        #cv2.imwrite(outputFile,image_aug)
    elif code==5:
        if verbose: print("Doing Data augmentation 5 (contrast) to image ")
        image_aug=iaa.LinearContrast((0.75, 1.5))(image=image)
        #cv2.imwrite(outputFile,image_aug)
    else:
        raise Exception("datasets.py Data augmentation technique not recognised ")
    return np.moveaxis(image_aug,-1,0)

# Function to determine if a patch contains floor 0, trees 1 or both 2 (mixed)
# counter there for debugging purposes

def patchList(im,numLabels,interesting,uninteresting,augment):

    #the target will be a list of 0 (class not in the patch) or 1 (class in the patch)
    # position i of the list contains info on label i+1
    totalPixels=im.size
    classThreshold=2*im.size/100
    result=[]
    result2=[]

    for x in range(numLabels):
        classPixels=np.sum(im==x+1)
        if classPixels>classThreshold:
            #we initialize with the probability!!!

            if x not in interesting:
                if x in uninteresting:result.append(min((2*classPixels/augment)/totalPixels,1))
                else:result.append(classPixels/totalPixels)
            else: result.append(min((2*augment*classPixels)/totalPixels,1))
            #print("appending "+str(result[-1]))
            #in the other result, just zeros and ones
            #result.append(classPixels/totalPixels)
            result2.append(1)
        else:
            result.append(0)
            result2.append(0)

    return np.array(result, dtype='f'),np.array(result2, dtype='f')

def uninterestingPatch(targ,unintClasses):
    #if we reach here we already know there is some class in this patch
    #print("entering "+str(targ)+" "+str(unintClasses))
    for i in range(len(targ[0])):
        #print("i"+str(i)+" targ[i] "+str(targ[i]))
        #if targ[i]==1 and i not in unintClasses:return False # this in not an uniteresting patch
        if targ[0][i]!=0 and i not in unintClasses:return False # this in not an uniteresting patch
    return True

def patchWithClasses(targ):
    for a in targ[0]:
        if a>0: return True
    return False

class Cropping2DDataset(Dataset):
    def __init__(
            self,
            data, labels, patch_size=32, overlap=16,nChan=3,augmentFactor=0,interestingClasses=[],uninterestingClasses=[],augment=False
    ):
        # Init
        self.data = data
        self.labels = labels
        self.numLabels=0
        for lab in labels:
            currentMax=np.max(lab)
            if currentMax>self.numLabels:
                self.numLabels=currentMax

        self.interesting=interestingClasses
        self.uninteresting=uninterestingClasses
        self.numAugmentations=6
        self.downSampleUninterestingPercentage=50
        self.AF=augmentFactor
        self.countInteresting=0
        self.countUninteresting=0

        #print(self.data)
        data_shape = self.data[0].shape

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)
        self.patch_size = patch_size
        self.overlap = overlap

        self.patch_slices = get_slices_bb(
            self.labels, self.patch_size, self.overlap
        )
        self.max_slice = np.cumsum(list(map(len, self.patch_slices)))

        # now perform data augmentation, go over the whole dataset
        # ignore patches with no labels!
        # for the patches from interesting classes, make a note to later make augmentFactor copies
        self.AugmDict={}
        self.realCount=0
        #uncount=0
        for i in range(self.max_slice[-1]):
            im,targ=self.accessRaw(i)
            #if 1 in targ:# at least some label present
            if patchWithClasses(targ):# at least some label present
                if (not uninterestingPatch(targ,self.uninteresting)) or random.randint(0,99)>self.downSampleUninterestingPercentage:#random draw
                    self.AugmDict[self.realCount]=i
                    self.realCount+=1
                    if uninterestingPatch(targ,self.uninteresting):self.countUninteresting+=1

                #else:
                #    print("ignoring unimportant patch "+str(uncount))
                #    uncount+=1

        print("now go to augment, real "+str(self.realCount)+" there were "+str(self.max_slice[-1])+" augment factor "+str(augmentFactor))

        if augment:
            augmCount=self.realCount
            for i in range(self.realCount):
                im,targDouble=self[i]
                targ=targDouble[0]
                # targ is a list of probabilities if the patch belongs to the class at that position
                patchToAugment=False
                superInterestingPatch=True
                labelsInPatch=0
                j=0
                while j<self.numLabels:
                    #print("targ "+str(targ[j]))
                    #if targ[j]==1 and j in self.interesting:
                    if targ[j]!=0 and j in self.interesting:
                       patchToAugment=True
                       labelsInPatch+=1
                    #elif targ[j]==1:
                    elif targ[j]!=0:
                        labelsInPatch+=1
                        superInterestingPatch=False
                    j+=1

                #if superInterestingPatch, double the augment factor
                if superInterestingPatch and (labelsInPatch!=0):
                    self.countInteresting+=1
                    #print("super interesting!!! "+str(targ)+" labels in patch "+str(labelsInPatch)+" "+str((labelsInPatch!=0)) )
                    for k in range(2*augmentFactor):
                        self.countInteresting+=1
                        self.AugmDict[augmCount]=i
                        augmCount+=1
                elif patchToAugment: #if not superinteresting but contains interesting, augment with the normal augment factor
                    self.countInteresting+=1
                    for k in range(augmentFactor):
                        self.countInteresting+=1
                        self.AugmDict[augmCount]=i
                        augmCount+=1


        print("FINISHED CREATING DATASET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(self.AugmDict)
        self.len=len(self.AugmDict)
        print("length was "+str(self.realCount)+" and will become "+str(self.len)+" interesting patches made "+str(100*self.countInteresting/self.realCount)+" uninteresting patches left "+str(100*self.countUninteresting/self.realCount))

    def accessRaw(self, index):
        if index<self.max_slice[-1]: #this part corresponds to the original images of the dataset
            # We select the case
            case_idx = np.min(np.where(self.max_slice > index))

            slices = [0] + self.max_slice.tolist()
            patch_idx = index - slices[case_idx]
            case_slices = self.patch_slices[case_idx]

            # We get the slice indexes
            none_slice = (slice(None, None),)
            slice_i = case_slices[patch_idx]

            inputs = self.data[case_idx][none_slice + slice_i].astype(np.float32)

            labels = self.labels[case_idx][slice_i].astype(np.uint8)

            #target=np.expand_dims(patchList(labels,self.numLabels),0)
            target=patchList(labels,self.numLabels,self.interesting,self.uninteresting,self.AF)

            return inputs, target
        else: raise Exception("wrong item access ")

    def __getitem__(self, index):
        #print("index "+str(index)+" real count "+str(self.realCount),flush=True)
        if index<self.realCount: #this part corresponds to the original images of the dataset

            # We select the case
            case_idx = np.min(np.where(self.max_slice > self.AugmDict[index]))

            slices = [0] + self.max_slice.tolist()
            #patch_idx = index - slices[case_idx]
            patch_idx = self.AugmDict[index] - slices[case_idx]
            case_slices = self.patch_slices[case_idx]

            # We get the slice indexes
            none_slice = (slice(None, None),)
            slice_i = case_slices[patch_idx]
            inputs = self.data[case_idx][none_slice + slice_i].astype(np.float32)

            labels = self.labels[case_idx][slice_i].astype(np.uint8)

            #target=np.expand_dims(patchList(labels,self.numLabels),0)
            target=patchList(labels,self.numLabels,self.interesting,self.uninteresting,self.AF)

            return inputs, target
        else: #in this part, we are augmenting the dataset
            # first, find the proper image that will be augmented

            original=self.AugmDict[index]
            im,targ=self[original]
            augmIm=augment(np.moveaxis(im,0,-1),random.randint(0, self.numAugmentations-1),False)

            return np.ascontiguousarray(augmIm),targ


    def __len__(self):
        return self.len
        #return self.max_slice[-1]
