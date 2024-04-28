import skimage

import numpy as np # computations in the mse command
from PIL import Image, ImageChops, ImageStat  # Image Library

import glob # Import file paths
import mahotas
import pylab
import random # shuffle array
import math
import sys
import os

THRESHOLD = .9

def compare_prints(imageA, imageB):
	imA = Image.open(imageA)
	imB = Image.open(imageB)
	stat1 = ImageStat.Stat(imA)
	stat2 = ImageStat.Stat(imB)
	difference = math.fabs(stat1.rms[0] - stat2.rms[0])/(stat1.rms[0])
	print(1-difference)
	return(1-difference)

def main(directory_1, directory_2):
	true_positive_count = 0
	true_negative_count = 0
	false_positive_count = 0
	false_negative_count = 0

	for directory in [directory_1, directory_2]:
    	file_list = os.listdir(directory)
    	f_files = sorted([f for f in file_list if f.startswith('f')and f.endswith('.png')])
    	s_files = sorted([s for s in file_list if s.startswith('s')and s.endswith('.png')])

    	reference_f_path = os.path.join(directory, f_files[0])

    	for f in f_files:
        	f_path = os.path.join(directory, f)

        	f_number = f.split('.')[0][1:]
        	s_filename = 's' + f_number + '.png'
        	s_path = os.path.join(directory, s_filename)

        	similarity_score_reference = compare_prints(f_path, reference_f_path)

        	if similarity_score_reference > THRESHOLD:
            	false_positive_count += 1
        	else:
            	true_negative_count += 1

        	similarity_score_corresponding = compare_prints(f_path, s_path)

        	if similarity_score_corresponding > THRESHOLD:
            	true_positive_count += 1
        	else:
            	false_negative_count += 1

	return true_positive_count, true_negative_count, false_positive_count, false_negative_count

if __name__ == "__main__":
	print(main(sys.argv[1], sys.argv[2]))
