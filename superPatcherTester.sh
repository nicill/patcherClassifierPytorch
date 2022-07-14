#!/bin/bash

size=100
it=10

dataDir=INTRODUEIX PATH


for channels in 3 
do
	for net in wideresnet dense vgg
	do
		echo "Starting test for network $net "
		date
		for frozen in False
		do				
			echo "python forestPatchClassifier.py -d $dataDir -arch $net -t $size -frozen $frozen -paramF paramsBlueBerry$size -e 10 -numC $channels>> outPatcher"size"$Size"channels"$channels"net"$net.txt"	
			python forestPatchClassifier.py -d $dataDir -arch $net -t $size -frozen $frozen -paramF paramsBlueBerry$size -e $it -B 8 -numC $channels >> SUPERAUGMoutPatcher"iterations"$it"size"$size"channels"$channels"net"$net.txt
		done
		echo "Ended test for network $net "
		date
	done
done

