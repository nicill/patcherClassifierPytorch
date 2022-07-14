# Example of the Paired Student's t-test
from scipy.stats import ttest_rel, ttest_ind
import numpy as np
# medium vs super
#data1 = [98.338663158,98.8402851788,98.7629834573,98.7192462079,98.3001922172,98.6478426018,98.5629122231,98.6251115949,98.5958285963,98.9472786495,98.2406824389,98.6069806511,98.7340828012,98.8286645435,98.8391121592,98.8469407519,98.8058247959,98.7262880462,98.9483829595,98.9040943812,98.9371452849,98.979812062,98.9904319687,98.9368997433,98.9121724382,99.0273087528,99.0181309266,98.7540111417,98.8261045299,98.8595625318,98.8738777112,98.9769091187,98.8784620833,98.9636982494,98.9366587428,98.9692866547]
#data2 = [98.6513981849,98.5097606879,98.2620347132,98.342432502,98.4017254236,98.1003194858,98.3512700847,98.2668233593,98.3806863773,98.7083075091,98.754290521,98.7514276088,98.6052102807,98.6822332411,98.629908289,98.3793784114,98.6152146899,98.5250433765,98.7511866082,98.7081098441,98.8397676877,98.726936145,98.6270402176,98.7859492715,98.6958326462,98.8558082497,98.7595382671,98.6710568463,98.8037874327,98.8193769427,98.8479363214,98.8397276567,98.8355233566,98.8876073081,98.8105465872,98.9263085152]
#stat=4.478, p=0.00007676, Probably different distributions


# random vs imagenet
#data1 = [98.6513981849,98.5097606879,98.2620347132,98.342432502,98.4017254236,98.1003194858,98.3512700847,98.2668233593,98.3806863773,98.7083075091,98.754290521,98.7514276088,98.6052102807,98.6822332411,98.629908289,98.3793784114,98.6152146899,98.5250433765]
#data2 = [98.1835915393,98.2983527386,98.4067973718,98.2942423197,98.3258775509,98.0603491074,98.3134162937,98.2629661141,98.4448820473,98.6313223092,98.3448447779,98.544398521,98.3185649994,98.2235728544,98.1729509957,98.5798949202,98.4807512912,98.2758650028]
#stat=3.163, p=0.00567828, Probably different distributions


# Alexnet vs resnet
#data1 = [91.4427412867,73.9850203835,75.2790015533,69.197905089,69.1894886433,75.2705851075,75.5720733778,75.8897817112,75.8897817112,97.8255141759,95.9705339246,94.8736694003,95.8072674087,96.4248989735,92.6278331704,95.6349972685,95.2532682143,94.8674968434,98.5493951506,98.6133163981,98.4116943427,98.5937899966,98.4296794208,98.3285722973,98.4116301741,98.2393402303,98.1943705298,98.4630592154,98.4857221309,98.5057625166,98.5181816747,98.5257044801,98.6260407253,98.6414973571,98.6530160634,98.6490797901]
#data2 = [98.6513981849,98.5097606879,98.2620347132,98.342432502,98.4017254236,98.1003194858,98.3512700847,98.2668233593,98.3806863773,98.7083075091,98.754290521,98.7514276088,98.6052102807,98.6822332411,98.629908289,98.3793784114,98.6152146899,98.5250433765,98.7511866082,98.7081098441,98.8397676877,98.726936145,98.6270402176,98.7859492715,98.6958326462,98.8558082497,98.7595382671,98.6710568463,98.8037874327,98.8193769427,98.8479363214,98.8397276567,98.8355233566,98.8876073081,98.8105465872,98.9263085152]
#stat=3.163, p=0.00567828, Probably different distributions

#resnet vs resbig
#data1 = [98.6513981849,98.5097606879,98.2620347132,98.342432502,98.4017254236,98.1003194858,98.3512700847,98.2668233593,98.3806863773,98.7083075091,98.754290521,98.7514276088,98.6052102807,98.6822332411,98.629908289,98.3793784114,98.6152146899,98.5250433765,98.7511866082,98.7081098441,98.8397676877,98.726936145]
#data2 = [98.4723686832,98.2054930745,98.4724644191,98.1116349521,98.1300564252,98.4757507389,98.3832024371,98.4771406215,98.3743636178,98.6742616542,98.3090840722,98.5559475581,98.6453558388,98.5571886691,98.2043573932,98.7262795825,98.3170147965,98.5072468986,98.1576198124,98.3591795539,98.4517330149,98.0889903966]


# is the repeat test with NOA different than that with SUPERAUGM?
#data1 = [98.7975096611,98.7538033607,98.4112890928,98.4364342367,98.022446001,98.6456987685,98.3786172334,97.7266024826,98.8328659548,98.2414551827,98.7029301893,98.6218764563,98.832208774,98.4541023711,98.8757737325,98.7813316799,98.5154809374,98.298246471,98.6453925653,98.756564962]
#data2 = [98.3767255507,98.4961284792,98.501419145,98.5210462392,98.2627131428,98.5437322583,98.4595694532,98.4295188809,98.2292332738,98.2689558402,98.2810997442,98.4164881481,98.3480219413,98.240251815,98.5294581439,98.2026272735,98.363646323,98.4417296397,97.4281116441,98.236781869]

#test2
#ALEXNET,91.4427412867,73.9850203835,75.2790015533,69.197905089,69.1894886433,75.2705851075,75.5720733778,75.8897817112,75.8897817112,97.8255141759,95.9705339246,94.8736694003,95.8072674087,96.4248989735,92.6278331704,95.6349972685,95.2532682143,94.8674968434,98.5493951506,98.6133163981,98.4116943427,98.5937899966,98.4296794208,98.3285722973,98.4116301741,98.2393402303,98.1943705298,98.4630592154,98.4857221309,98.5057625166,98.5181816747,98.5257044801,98.6260407253,98.6414973571,98.6530160634,98.6490797901
#SQUEEZENET,88.9430801823,96.9852320366,76.283721733,68.8548120501,70.2766870501,68.8548120501,68.8548120501,77.8646172094,68.8548120501,98.2301533222,98.477300526,98.5326223025,98.2651373895,96.9031507791,96.3977517476,97.8922715054,97.1308082907,88.4911045254,98.7093841747,98.4938090675,98.8307931016,98.76574217,98.5595747395,98.7177983499,98.7688128632,98.3145076163,94.5220475861,97.5906947156,98.5980021423,98.7364591713,98.8915739122,98.78036933,98.7576748471,98.7194224216,98.70224433,98.6729258413
#ResNet50,98.6513981849,98.5097606879,98.2620347132,98.342432502,98.4017254236,98.1003194858,98.3512700847,98.2668233593,98.3806863773,98.7083075091,98.754290521,98.7514276088,98.6052102807,98.6822332411,98.629908289,98.3793784114,98.6152146899,98.5250433765,98.7511866082,98.7081098441,98.8397676877,98.726936145,98.6270402176,98.7859492715,98.6958326462,98.8558082497,98.7595382671,98.6710568463,98.8037874327,98.8193769427,98.8479363214,98.8397276567,98.8355233566,98.8876073081,98.8105465872,98.9263085152
#ResNet152,98.4723686832,98.2054930745,98.4724644191,98.1116349521,98.1300564252,98.4757507389,98.3832024371,98.4771406215,98.3743636178,98.6742616542,98.3090840722,98.5559475581,98.6453558388,98.5571886691,98.2043573932,98.7262795825,98.3170147965,98.5072468986,98.1576198124,98.3591795539,98.4517330149,98.0889903966,98.7256591282,98.2867290012,98.6992739136,97.9678210141,98.730419714,98.7161394064,98.6073765994,98.6887220983,98.8009639333,98.9094733532,98.7870242848,98.8501133743,98.8648750648,98.87
#VGG,78.4327272965,68.8652287168,76.2107856225,68.8548120501,68.8548120501,68.8548120501,68.8548120501,68.8548120501,68.8884778331,98.4471577248,98.1643428649,96.2391499217,94.6080127275,87.6401879351,86.6007457586,89.4341955718,75.9498758047,82.66692583,98.6459156315,98.6192483513,98.4631650673,98.4044870249,98.4897935529,98.2342265989,98.6947863114,98.4752322137,98.3618557378,98.8196860346,98.9032649094,98.8204119249,98.8434910207,98.7622936752,98.7742966531,98.6727535504,98.7027178673,98.660748302
#densenet,98.5214426032,98.3850280966,98.5307995318,98.2812076641,98.1361945072,98.313281145,98.4068374028,98.4990388521,98.5269775741,98.4911285451,98.4461243948,98.6099708665,98.4504941744,98.3965990007,98.6359089518,98.5743954393,98.5708850465,98.3703706057,98.9802237013,98.6597042378,98.6760844249,98.7516979061,98.6690217536,98.6005813366,98.5254498565,98.4669119196,98.5416016321,98.7465599346,98.7951299864,98.8744668007,98.7845684709,98.9162370487,98.9365883809,98.8564563485,98.852730745,98.8618043715
#ResNext,98.5231112469,98.4408444631,98.1148098449,98.0897969673,98.3312441364,98.1542633504,98.429184415,98.2892085478,98.2893438989,98.3864743023,98.6420154665,98.1263423703,98.5747068017,98.6614989482,98.6500495698,98.6315581506,98.7608418945,98.6122746043,98.8196821118,98.5492406015,98.6372777818,98.5976726159,98.5391478862,98.1362712584,98.1881478448,98.626802724,98.6350164161,98.8548035983,98.8342828638,98.8830864864,98.8492129225,98.8207239056,98.8984378845,98.6265600712,98.7951964255,98.7836067392
#wideResNet,98.4227000988,98.7454950373,98.451638313,89.9005159806,98.199184972,98.3551910827,98.2053579258,98.591609021,98.1338552793,98.3631313049,97.4333802278,98.5468429825,98.4901736249,97.5687861602,98.4237018615,97.8410576749,98.5872679199,98.5621553837,98.8416985808,98.4829543364,98.3546019933,97.7604553988,98.4745709077,98.4803311882,97.1661200468,98.3199315589,98.7894035437,98.7453891855,98.6075668519,98.8300053131,98.7361729805,98.8633265141,98.691295931,98.9343133217,98.8050920941,98.8127494299

#resnext vs resnet50
data=[["ALEXNET",91.4427412867,73.9850203835,75.2790015533,69.197905089,69.1894886433,75.2705851075,75.5720733778,75.8897817112,75.8897817112,97.8255141759,95.9705339246,94.8736694003,95.8072674087,96.4248989735,92.6278331704,95.6349972685,95.2532682143,94.8674968434,98.5493951506,98.6133163981,98.4116943427,98.5937899966,98.4296794208,98.3285722973,98.4116301741,98.2393402303,98.1943705298,98.4630592154,98.4857221309,98.5057625166,98.5181816747,98.5257044801,98.6260407253,98.6414973571,98.6530160634,98.6490797901], 
["SQUEEZENET",88.9430801823,96.9852320366,76.283721733,68.8548120501,70.2766870501,68.8548120501,68.8548120501,77.8646172094,68.8548120501,98.2301533222,98.477300526,98.5326223025,98.2651373895,96.9031507791,96.3977517476,97.8922715054,97.1308082907,88.4911045254,98.7093841747,98.4938090675,98.8307931016,98.76574217,98.5595747395,98.7177983499,98.7688128632,98.3145076163,94.5220475861,97.5906947156,98.5980021423,98.7364591713,98.8915739122,98.78036933,98.7576748471,98.7194224216,98.70224433,98.6729258413],
["ResNet50",98.6513981849,98.5097606879,98.2620347132,98.342432502,98.4017254236,98.1003194858,98.3512700847,98.2668233593,98.3806863773,98.7083075091,98.754290521,98.7514276088,98.6052102807,98.6822332411,98.629908289,98.3793784114,98.6152146899,98.5250433765,98.7511866082,98.7081098441,98.8397676877,98.726936145,98.6270402176,98.7859492715,98.6958326462,98.8558082497,98.7595382671,98.6710568463,98.8037874327,98.8193769427,98.8479363214,98.8397276567,98.8355233566,98.8876073081,98.8105465872,98.9263085152],
["ResNet152",98.4723686832,98.2054930745,98.4724644191,98.1116349521,98.1300564252,98.4757507389,98.3832024371,98.4771406215,98.3743636178,98.6742616542,98.3090840722,98.5559475581,98.6453558388,98.5571886691,98.2043573932,98.7262795825,98.3170147965,98.5072468986,98.1576198124,98.3591795539,98.4517330149,98.0889903966,98.7256591282,98.2867290012,98.6992739136,97.9678210141,98.730419714,98.7161394064,98.6073765994,98.6887220983,98.8009639333,98.9094733532,98.7870242848,98.8501133743,98.8648750648,98.87],
["VGG",78.4327272965,68.8652287168,76.2107856225,68.8548120501,68.8548120501,68.8548120501,68.8548120501,68.8548120501,68.8884778331,98.4471577248,98.1643428649,96.2391499217,94.6080127275,87.6401879351,86.6007457586,89.4341955718,75.9498758047,82.66692583,98.6459156315,98.6192483513,98.4631650673,98.4044870249,98.4897935529,98.2342265989,98.6947863114,98.4752322137,98.3618557378,98.8196860346,98.9032649094,98.8204119249,98.8434910207,98.7622936752,98.7742966531,98.6727535504,98.7027178673,98.660748302],
["densenet",98.5214426032,98.3850280966,98.5307995318,98.2812076641,98.1361945072,98.313281145,98.4068374028,98.4990388521,98.5269775741,98.4911285451,98.4461243948,98.6099708665,98.4504941744,98.3965990007,98.6359089518,98.5743954393,98.5708850465,98.3703706057,98.9802237013,98.6597042378,98.6760844249,98.7516979061,98.6690217536,98.6005813366,98.5254498565,98.4669119196,98.5416016321,98.7465599346,98.7951299864,98.8744668007,98.7845684709,98.9162370487,98.9365883809,98.8564563485,98.852730745,98.8618043715],
["ResNext",98.5231112469,98.4408444631,98.1148098449,98.0897969673,98.3312441364,98.1542633504,98.429184415,98.2892085478,98.2893438989,98.3864743023,98.6420154665,98.1263423703,98.5747068017,98.6614989482,98.6500495698,98.6315581506,98.7608418945,98.6122746043,98.8196821118,98.5492406015,98.6372777818,98.5976726159,98.5391478862,98.1362712584,98.1881478448,98.626802724,98.6350164161,98.8548035983,98.8342828638,98.8830864864,98.8492129225,98.8207239056,98.8984378845,98.6265600712,98.7951964255,98.7836067392],
["wideResNet",98.4227000988,98.7454950373,98.451638313,89.9005159806,98.199184972,98.3551910827,98.2053579258,98.591609021,98.1338552793,98.3631313049,97.4333802278,98.5468429825,98.4901736249,97.5687861602,98.4237018615,97.8410576749,98.5872679199,98.5621553837,98.8416985808,98.4829543364,98.3546019933,97.7604553988,98.4745709077,98.4803311882,97.1661200468,98.3199315589,98.7894035437,98.7453891855,98.6075668519,98.8300053131,98.7361729805,98.8633265141,98.691295931,98.9343133217,98.8050920941,98.8127494299]]

for i in range(len(data)):
    for j in range(i+1, len(data)):
        data1=data[i][1:]
        data2=data[j][1:]

        stat, p = ttest_rel(data1, data2)
        #stat, p = ttest_ind(data1, data2)
        print('stat=%.3f, p=%.8f' % (stat, p))
        av1=np.mean(data1)
        stdev1=np.std(data1)
        av2=np.mean(data2)
        stdev2=np.std(data2)
        if p > 0.05:
            print('Probably the same distribution '+str(data[i][0])+" "+str(av1)+" +- "+str(stdev1)+" and "+str(data[j][0])+" "+str(av2)+" +- "+str(stdev2))
        else:
            print('Probably different distributions '+str(data[i][0])+" "+str(av1)+" +- "+str(stdev1)+" and "+str(data[j][0])+" "+str(av2)+" +- "+str(stdev2))
