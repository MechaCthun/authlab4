#import pandas as pd
import glob 
#import progressbar
import cv2
from matplotlib import pyplot as plt
from PIL import Image,ImageEnhance
import numpy as np
import os as os

def cleanup_img(path):
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    return(cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 181, 11))

def find_minutae(path, n=150, disp=False):
    img = cleanup_img(path)
    dst = cv2.cornerHarris(img,6,5,0.04)
    thresh = sorted(dst.flatten(), reverse=True)[n-1]
    minutae = np.array(np.where(dst > thresh))
    if disp: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img[dst>thresh]=[255, 127, 0]
        plt.figure(figsize=(12,12))
        plt.imshow(img)
    return(minutae)


def centroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def compare_prints(path_a, path_b, thresh = 110, debug=False):
    m_a = find_minutae(path_a, disp=debug)
    m_b = find_minutae(path_b, disp=debug)
    c_a = np.expand_dims(centroidnp(m_a), 0)
    c_b = np.expand_dims(centroidnp(m_b), 0)
    ca_test = c_a[:, None]
    dists_a = np.linalg.norm(np.transpose(m_a) - c_a[:, None], axis=2)
    dists_b = np.linalg.norm(np.transpose(m_b) - c_b[:, None], axis=2)
    #print(len(sorted(dists_a[0])), len(sorted(dists_b[0])))
    try:
        sort_dists = np.array(sorted(dists_a[0])) - np.array(sorted(dists_b[0]))      
    except Exception:
        return 0
    
    similarity = len(sort_dists[np.where(abs(sort_dists) < thresh)]) / len(sort_dists)

    return similarity





def main():
    path_fingerprint = 'C:\\Users\\Trubber\\Documents\\auth\\TestF'
    path_sFringerprint = 'C:\\Users\\Trubber\\Documents\\auth\\TestS'
    file_paths_f = []
    file_paths_s = []

    #load files for comparison 
    for file in os.listdir(path_fingerprint):
        if file.endswith('.png'):
            file_paths_f.append(path_fingerprint + "\\" + os.path.basename(file))
        elif file.endswith('.txt'):
            pass
        else:
            print("file loading error")

    for file in os.listdir(path_sFringerprint):
        if file.endswith('.png'):
            file_paths_s.append(path_sFringerprint + "\\" + os.path.basename(file))
        elif file.endswith('.txt'):
            pass
        else:
            print("file loading error")

    
    true_positive_count = 0
    true_negative_count = 0

    false_negative_count = 0
    false_positive_count = 0

    for x in range(len(file_paths_f)):
        similarity = compare_prints(file_paths_f[x], file_paths_s[x])

    
        if similarity >= .9:
            true_positive_count += 1
        else:
            false_negative_count += 1


        similarity, eer, far, frr = compare_prints(file_paths_f[x], file_paths_s[0])

       
        if similarity >= .9:
            false_positive_count += 1
        else:
            true_negative_count += 1
        
        
    
    print("\n TP, TN, FP, FN:")
    print(f"{true_positive_count}, {true_negative_count}, {false_positive_count},  {false_negative_count}")


main()