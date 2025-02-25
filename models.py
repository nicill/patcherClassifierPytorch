import itertools
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from data_manipulation.models import BaseModel
from data_manipulation.utils import to_torch_var, time_to_string
from torchvision import transforms, models

import cv2
import sys

def uselessImage(im):
    totalPixels=im.size
    th=95*im.size/100
    return np.sum(im<5)>th or np.sum(im>250)>th

def dsc_loss(pred, target, smooth=0.1):
    """
    Loss function based on a single class DSC metric.
    :param pred: Predicted values. This tensor should have the shape:
     [batch_size, n_classes, data_shape]
    :param target: Ground truth values. This tensor can have multiple shapes:
     - [batch_size, n_classes, data_shape]: This is the expected output since
       it matches with the predicted tensor.
     - [batch_size, data_shape]: In this case, the tensor is labeled with
       values ranging from 0 to n_classes. We need to convert it to
       categorical.
    :param smooth: Parameter used to smooth the DSC when there are no positive
     samples.
    :return: The mean DSC for the batch
    """
    dims = pred.shape
    assert target.shape == pred.shape,\
        'Sizes between predicted and target do not match'
    target = target.type_as(pred)

    reduce_dims = tuple(range(1, len(dims)))
    num = (2 * torch.sum(pred * target, dim=reduce_dims))
    den = torch.sum(pred + target, dim=reduce_dims) + smooth
    dsc_k = num / den
    dsc = 1 - torch.mean(dsc_k)

    return torch.clamp(dsc, 0., 1.)

class myFeatureExtractor(BaseModel):
    def __init__(self,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    n_outputs=3,frozen=True,featEx="res",LR=1e-1,nChanInit=3):
        super().__init__()
        # Init values
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.n_outputs=n_outputs
        self.device = device
        self.weights=torch.Tensor([8,2,2,1,2,2])
        self.norm = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        print("Creating myFeatureExtractor with device "+str(self.device))

        # add a convolution to move from 4 to 3 channels
        self.n_image_channels = nChanInit
        self.chanT = nn.Conv2d(self.n_image_channels,
                                       3,
                                       kernel_size=3,
                                       padding=1)

        #now add a feature extractor, CHECK THIS!
        if featEx == "res":
            weights = models.ResNet50_Weights.DEFAULT
            self.featEx = models.resnet50(weights=weights)
            # change final layer
            num_ftrs = self.featEx.fc.in_features
            self.featEx.fc = nn.Linear(num_ftrs, 2*self.n_outputs)

            # used to be 
            #self.featEx = torchModels.resnet50(pretrained=True)
            #num_ftrs = self.featEx.fc.in_features
            #self.featEx.fc = nn.Linear(num_ftrs, 2*self.n_outputs)
        elif featEx == "resBIG":
            print("careful, untested")
            weights = models.ResNet152_Weights.DEFAULT
            self.featEx = models.resnet152(weights=weights)
            num_ftrs = self.featEx.fc.in_features
            self.featEx.fc = nn.Linear(num_ftrs, 2*self.n_outputs)
        elif arch == "swimt":
            weights = models.Swin_T_Weights.DEFAULT
            self.featEx = models.swin_t(weights=weights)
            num_ftrs = self.featEx.head.in_features
            self.featEx.head = nn.Linear(num_ftrs, 2*self.n_outputs

        elif arch == "swims":
            weights = models.Swin_S_Weights.DEFAULT
            self.featEx = models.swin_s(weights=weights)
            num_ftrs = self.featEx.head.in_features
            self.featEx.head = nn.Linear(num_ftrs, 2*self.n_outputs

        elif arch == "swimb":
            weights = models.Swin_B_Weights.DEFAULT
            self.featEx = models.swin_b(weights=weights)
            # swim B needs smaller batch size because of memory requirements
            bs = 8
            num_ftrs = self.featEx.head.in_features
            self.featEx.head = nn.Linear(num_ftrs, 2*self.n_outputs
        elif featEx == "vitb16":
            weights = models.ViT_B_16_Weights.DEFAULT
            self.featEx = models.vit_b_16(weights=weights)
            num_ftrs = self.featEx.heads[0].in_features
            self.featEx.heads[0] = nn.Linear(num_ftrs, 2*self.n_outputs
        elif featEx == "vitb32":
            weights = models.ViT_B_32_Weights.DEFAULT
            self.featEx = models.vit_b_32(weights=weights)
            num_ftrs = self.featEx.heads[0].in_features
            self.featEx.heads[0] = nn.Linear(num_ftrs, 2*self.n_outputs
        elif featEx == "vith14":
            weights = models.ViT_H_14_Weights.DEFAULT
            self.featEx = models.vit_h_14(weights=weights)
            num_ftrs = self.featEx.heads[0].in_features
            self.featEx.heads[0] = nn.Linear(num_ftrs, 2*self.n_outputs
        elif featEx == "vitl16":
            weights = models.ViT_L_16_Weights.DEFAULT
            self.featEx = models.vit_l_16(weights=weights)
            num_ftrs = self.featEx.heads[0].in_features
            self.featEx.heads[0] = nn.Linear(num_ftrs, 2*self.n_outputs
        elif featEx == "vitl32":
            weights = models.ViT_L_32_Weights.DEFAULT
            self.featEx = models.vit_l_32(weights=weights)
            num_ftrs = self.featEx.heads[0].in_features
            self.featEx.heads[0] = nn.Linear(num_ftrs, 2*self.n_outputs
        elif featEx == "convnexts":
            weights = models.convnext.ConvNeXt_Small_Weights.DEFAULT
            self.featEx = models.convnext_small(weights=weights)
            num_ftrs = self.featEx.classifier[-1].in_features
            self.featEx.classifier[-1] = nn.Linear(num_ftrs, outputNumClasses)
        elif featEx == "convnextt":
            weights = models.convnext.ConvNeXt_Tiny_Weights.DEFAULT
            self.featEx = models.convnext_tiny(weights=weights)
            num_ftrs = self.featEx.classifier[-1].in_features
            self.featEx.classifier[-1] = nn.Linear(num_ftrs, outputNumClasses)
        elif featEx == "convnextb":
            weights = models.convnext.ConvNeXt_Base_Weights.DEFAULT
            self.featEx = models.convnext_base(weights=weights)
            bs = 8
            num_ftrs = self.featEx.classifier[-1].in_features
            self.featEx.classifier[-1] = nn.Linear(num_ftrs, outputNumClasses)
        elif featEx == "convnextl":
            weights = models.convnext.ConvNeXt_Large_Weights.DEFAULT
            self.featEx = models.convnext_large(weights=weights)
            bs = 8
            num_ftrs = self.featEx.classifier[-1].in_features
            self.featEx.classifier[-1] = nn.Linear(num_ftrs, outputNumClasses)
        elif featEx=="resnext":
            print("careful, untested")
            weights = models.ResNeXt101_32X8D_Weights.DEFAULT
            self.featEx = models.resnext101_32x8d(weights=weights)
            num_ftrs = self.featEx.fc.in_features
            self.featEx.fc = nn.Linear(num_ftrs, 2*self.n_outputs)
        elif featEx=="wideresnet":
            print("careful, untested")
            weights = models.Wide_ResNet101_2_Weights.DEFAULT
            self.featEx = models.wide_resnet101_2(weights=weights)
            num_ftrs = self.featEx.fc.in_features
            self.featEx.fc = nn.Linear(num_ftrs, 2*self.n_outputs)
        else: raise Exception("Exception when constructing feature extractor, unknown architecture "+str(featEx))


        # for others see https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

        self.chanT = self.chanT.to(device)
        self.featEx = self.featEx.to(device)

        # <Loss function setup>
        #nn.CrossEntropyLoss()
        self.train_functions = [
            {
                'name': 'xentrProb',
                'weight': 0.5,
                #'f': lambda p, t: F.binary_cross_entropy(p[0], t[0].type_as(p[0]).to(p[0].device),weight=self.weights.to(device))
                'f': lambda p, t: F.binary_cross_entropy(p[0], t[0].type_as(p[0]).to(p[0].device))
                #'f': lambda p, t: F.binary_cross_entropy(p[:, 1, ...],torch.squeeze(t, dim=1).type_as(p).to(p.device))
                #'f': lambda p, t: F.binary_cross_entropy(p[:, 1, ...],torch.squeeze(t, dim=1).type_as(p).to(p.device))
                #'f': lambda p, t: nn.CrossEntropyLoss(p,t)
            },
            {
                'name': 'xentr01',
                'weight': 1,
                #'f': lambda p, t: F.binary_cross_entropy(p[1], t[1].type_as(p[1]).to(p[1].device),weight=self.weights.to(device))
                'f': lambda p, t: F.binary_cross_entropy(p[1], t[1].type_as(p[1]).to(p[1].device))
                #'f': lambda p, t: F.binary_cross_entropy(p[:, 1, ...],torch.squeeze(t, dim=1).type_as(p).to(p.device))
                #'f': lambda p, t: F.binary_cross_entropy(p[:, 1, ...],torch.squeeze(t, dim=1).type_as(p).to(p.device))
                #'f': lambda p, t: nn.CrossEntropyLoss(p,t)
            },

        ]
        self.val_functions = [
            {
                'name': 'xeProb',
                'weight': 0.5,
                'f': lambda p, t: F.binary_cross_entropy(p[0], t[0].type_as(p[0]).to(p[0].device),weight=self.weights.to(device))
                #'f': lambda p, t: nn.CrossEntropyLoss(p,t)
                #'f': lambda p, t: F.binary_cross_entropy(p[:, 1, ...],torch.squeeze(t, dim=1).type_as(p).to(p.device))
            },
            {
                'name': 'xe01',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(p[1], t[1].type_as(p[1]).to(p[1].device),weight=self.weights.to(device))
                #'f': lambda p, t: nn.CrossEntropyLoss(p,t)
                #'f': lambda p, t: F.binary_cross_entropy(p[:, 1, ...],torch.squeeze(t, dim=1).type_as(p).to(p.device))
            },
        ]

        # <Optimizer setup>
        # We do this last setp after all parameters are defined
        #allParams=list(self.parameters())

        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=LR)

        if frozen:
            for param in self.featEx.parameters():

                param.requires_grad = False

            #now Unfreeze classifier part
            if featEx in ["res","resBIG","resnext","wideresnet"]:
                for param in self.featEx.fc.parameters():
                    param.requires_grad = True
            elif featEx=="dense":
                for param in self.featEx.classifier:
                    param.requires_grad = True
            elif featEx in ["vgg","alex"]:
                for param in self.featEx.classifier[6]:
                    param.requires_grad = True
            elif featEx=="squeeze":
                for param in self.featEx.classifier[1]:
                    param.requires_grad = True
            else: raise Exception("Exception when unfreezing feature extractor classifier, unknown architecture "+str(featEx))



        #initially, do not define scheduler
        self.scheduler =None

        # self.autoencoder.dropout = 0.99
        # self.dropout = 0.99
        # self.ann_rate = 1e-2

    def addScheduler(self,LR,steps,ep):
        #define also LR scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer_alg, max_lr=LR, steps_per_epoch=steps, epochs=ep)

    def forward(self, input_s):
        if self.n_image_channels==4:
            input_s = self.chanT(input_s)
        elif self.n_image_channels!=3:
            raise (Exception("Number of channels must either be 3 or 4"))

        out = []
        for x_ in input_s:
            out.append(self.norm(x_.cpu()))
        out = torch.stack(out)
        firstOutput = self.featEx(out.to(self.device))[:,:self.n_outputs,...]
        secondOutput = self.featEx(out.to(self.device))[:,self.n_outputs:,...]
        #return torch.softmax(firstOutput,dim=1),torch.sigmoid(secondOutput)
        return torch.sigmoid(firstOutput),torch.sigmoid(secondOutput)

    #def dropout_update(self):
    #    super().dropout_update()
    #    self.autoencoder.dropout = self.dropout

    def batch_update(self, epochs):
        self.scheduler.step()

        #return None


    def test(self, data, patch_size=50, verbose=True, refine=False,classesToRefine=[]):
            # Init
            self.eval()
            seg=[]
            for x in range(self.n_outputs):seg.append([])

            # Init
            t_in = time.time()

            for i, im in enumerate(data):

                # Case init
                t_case_in = time.time()

                # This branch is only used when images are too big. In this case
                # they are split in patches and each patch is trained separately.
                # Currently, the image is partitioned in blocks with no overlap,
                # however, it might be a good idea to sample all possible patches,
                # test them, and average the results. I know both approaches
                # produce unwanted artifacts, so I don't know.
                seg_i=[]
                if patch_size is not None:

                    # Initial results. Filled to 0.
                    for x in range(self.n_outputs):
                        #seg_i.append([np.zeros(im.shape[1:])])
                        seg_i.append(np.zeros(im.shape[1:]))

                    print("after initial filling, seg_i length "+str(len(seg_i)))

                    limits = tuple(
                        list(range(0, lim, patch_size))[:-1] + [lim - patch_size]
                        for lim in im.shape[1:] #was for lim in data.shape[1:]
                    )
                    print("limits out "+str(limits))
                    limits_product = list(itertools.product(*limits))

                    n_patches = len(limits_product)

                    # The following code is just a normal test loop with all the
                    # previously computed patches.
                    for pi, (xi, xj) in enumerate(limits_product):
                        # Here we just take the current patch defined by its slice
                        # in the x and y axes. Then we convert it into a torch
                        # tensor for testing.
                        xslice = slice(xi, xi + patch_size)
                        yslice = slice(xj, xj + patch_size)

                        currentImage=im[slice(None), xslice, yslice]

                        if not uselessImage(currentImage):

                            data_tensor = to_torch_var(
                                np.expand_dims(currentImage, axis=0)
                            )

                            # Testing itself.
                            with torch.no_grad():
                                #torch.cuda.synchronize()
                                #seg_pi, unc_pi, _, tops_pi = self(data_tensor)
                                seg_pi = self(data_tensor)
                                #print("seg_pi is "+str(seg_pi))
                                seg_piFirst=seg_pi[0][0]
                                #print("seg_piFirst is "+str(seg_piFirst))
                                seg_piSecond=seg_pi[1][0]
                                #print("seg_piSecond is "+str(seg_piSecond))
                                #torch.cuda.synchronize()
                                #torch.cuda.empty_cache()

                                #print("seg_pi is "+str(seg_pi))
                                #for au in seg_pi[0]:
                                #    if au>0.5 or au<0.5: print(au)
                            probThreshold=0.4
                            for x in range(self.n_outputs):
                                # we store the probability of this class in this patch
                                #seg_i[x][xslice, yslice]=int(255*seg_piFirst.cpu().numpy()[x]*seg_piSecond.cpu().numpy()[x])
                                if seg_piFirst.cpu().numpy()[x]>probThreshold:
                                    if not refine or (x not in classesToRefine) :seg_i[x][xslice, yslice]=255
                                    else: seg_i[x][xslice, yslice]=self.refine(currentImage,patch_size,x)

                            # Printing
                            #init_c = '\033[0m' if self.training else '\033[38;5;238m'
                            #whites = ' '.join([''] * 12)
                            #percent = 20 * (pi + 1) // n_patches
                            #progress_s = ''.join(['-'] * percent)
                            #remainder_s = ''.join([' '] * (20 - percent))

                            #t_out = time.time() - t_in
                            #t_case_out = time.time() - t_case_in
                            #time_s = time_to_string(t_out)

                            #t_eta = (t_case_out / (pi + 1)) * (n_patches - (pi + 1))
                            #eta_s = time_to_string(t_eta)
                            #batch_s = '{:}Case {:03}/{:03} ({:03d}/{:03d})' \
                            #          ' [{:}>{:}] {:} ETA: {:}'.format(
                            #    init_c + whites, i + 1, len(data), pi + 1, n_patches,
                            #    progress_s, remainder_s, time_s, eta_s + '\033[0m'
                            #)
                            #if verbose:
                            #    print('\033[K', end='', flush=True)
                            #    print(batch_s, end='\r', flush=True)
                        #else:
                        #    print("THIS WAS A USELESS IMAGE")

                if verbose:
                    print(
                        '\033[K%sSegmentation finished' % ' '.join([''] * 12)
                    )

                for x in range(self.n_outputs):seg[x].append(seg_i[x])
                #for x in range(self.n_outputs):print(str(len(seg_i[x]))+" ",end="")


            return seg


    #refine a mask created with patch predictions
    # return a binary mask with the same size as im with 255 where the class is present
    def refine(self, im, patch_size,c):
        #print("refining!"+str(c))
        #returnMask=np.zeros(im.shape[1:])
        #data_tensor = to_torch_var(np.expand_dims(im, axis=0))

        # make smaller images
        newPatchSize=patch_size//4

        # Initial results. Filled to 0.
        seg_i=[]
        for x in range(self.n_outputs):
            #seg_i.append([np.zeros(im.shape[1:])])
            seg_i.append(np.zeros(im.shape[1:]))

        #print("after initial filling, seg_i length "+str(len(seg_i)))

        limits = tuple(
            list(range(0, lim, newPatchSize))[:-1] + [lim - newPatchSize]
            for lim in im.shape[1:] #was for lim in data.shape[1:]
        )
        #print(limits)

        limits_product = list(itertools.product(*limits))
        #print(limits_product)

        n_patches = len(limits_product)

        # The following code is just a normal test loop with all the
        # previously computed patches.
        for pi, (xi, xj) in enumerate(limits_product):
            # Here we just take the current patch defined by its slice
            # in the x and y axes. Then we convert it into a torch
            # tensor for testing.
            xslice = slice(xi, xi + newPatchSize)
            yslice = slice(xj, xj + newPatchSize)

            currentImage=im[slice(None), xslice, yslice]


            if not uselessImage(currentImage):

                data_tensor = to_torch_var(np.expand_dims(currentImage, axis=0))

                # Testing itself.
                with torch.no_grad():
                    seg_pi = self(data_tensor)
                    seg_piFirst=seg_pi[0][0]
                    seg_piSecond=seg_pi[1][0]

                probThreshold=0.4

                # we store the probability of this class in this patch
                #seg_i[x][xslice, yslice]=int(255*seg_piFirst.cpu().numpy()[x]*seg_piSecond.cpu().numpy()[x])
                if seg_piFirst.cpu().numpy()[c]>probThreshold:
                    #print(seg_i[c][xslice, yslice].shape)    
                    seg_i[c][xslice, yslice]=255

        #print(seg_i[c].shape)

        return seg_i[c]
