import random
import os
import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

def cleanup_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    img = cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 181, 11)
    return img

def find_minutae(path, disp=False):
    print "Starting cleanup for image: {}".format(path)
    img = cleanup_img(path)
    print "Image cleanup completed. Proceeding to skeletonization."
    timg = img // 255
    img = skeletonize(timg)  # Ensure this is compatible with your skimage version
    print "Skeletonization completed. Calculating center of mass."
    com = ndimage.measurements.center_of_mass(img)
    print "Center of mass found at:", com
    cmask = create_circular_mask(512, 512, com, 224)
    print "Circular mask applied."
    img[cmask == 0] = 0
    if disp:
        plt.imshow(255 - img, 'gray')
        plt.show()

    stepSize = 3
    (w_width, w_height) = (3, 3)  # window size
    coords = []
    print "Starting minutiae extraction process."
    for x in range(0, img.shape[1] - w_width, stepSize):
        for y in range(0, img.shape[0] - w_height, stepSize):
            window = img[x:x + w_width, y:y + w_height]
            winmean = np.mean(window)
            if winmean in (8 / 9, 1 / 9):
                coords.append((x, y))
    print "Found {} candidate minutiae points.".format(len(coords))
    coords = np.array(coords)
    coords_centr = centroidnp(coords)
    sort_coords = sorted(coords, key=lambda coord: np.linalg.norm(coord - coords_centr))
    print "Minutiae points sorted."
    return np.array(sort_coords[1:100])

def centroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length

def compare_prints(path_a, path_b, thresh=5, debug=False):
    m_a = find_minutae(path_a, disp=debug)
    m_b = find_minutae(path_b, disp=debug)
    c_a = np.expand_dims(centroidnp(m_a), 0)
    c_b = np.expand_dims(centroidnp(m_b), 0)
    dists_a = np.linalg.norm(m_a - c_a[:, ], axis=1)
    dists_b = np.linalg.norm(m_b - c_b[:, ], axis=1)
    sort_dists = np.array(dists_a) - np.array(dists_b)
    similarity = len(sort_dists[np.where(abs(sort_dists) < thresh)]) / len(sort_dists)
    print similarity
    return similarity

def calculate_statistics(similarity_scores, threshold=0.5):
    true_accepts = [s for s in similarity_scores if s >= threshold]
    false_rejects = [s for s in similarity_scores if s < threshold]
    true_accept_rate = len(true_accepts) / len(similarity_scores)
    false_reject_rate = len(false_rejects) / len(similarity_scores)

    # Find the Equal Error Rate (EER) where FAR = FRR
    far = frr = 0
    for t in np.linspace(0, 1, 100):
        far_t = len([s for s in similarity_scores if s < t]) / len(similarity_scores)
        frr_t = len([s for s in similarity_scores if s >= t]) / len(similarity_scores)
        if abs(far_t - frr_t) < abs(far - frr):
            far, frr, threshold = far_t, frr_t, t

    statistics = {
        'FAR Min': min(far, frr),
        'FAR Max': max(far, frr),
        'FAR Average': np.mean([far, frr]),
        'FRR Min': min(far, frr),
        'FRR Max': max(far, frr),
        'FRR Average': np.mean([far, frr]),
        'Equal Error Rate': (far + frr) / 2
    }
    return statistics

def main(directory_1, directory_2):
    # Initialize counts for True Positive, True Negative, False Positive, and False Negative
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0

    # Loop through each directory
    for directory in [directory_1, directory_2]:
        # Get the list of files in the directory
        file_list = os.listdir(directory)
        # Filter out 'f' files and sort them
        f_files = sorted([f for f in file_list if f.startswith('f') and f.endswith('.png')])
        # Filter out 's' files and sort them
        s_files = sorted([s for s in file_list if s.startswith('s') and s.endswith('.png')])
        
        # Use the first 'f' fingerprint as the reference fingerprint
        reference_f_path = os.path.join(directory, f_files[0])
        
        # Iterate through each 'f' fingerprint file
        for f in f_files:
            # Construct the file path for 'f'
            f_path = os.path.join(directory, f)
            
            # Extract the number from the filename to find the corresponding 's' fingerprint
            f_number = f.split('.')[0][1:]
            s_filename = 's' + f_number + '.png'
            s_path = os.path.join(directory, s_filename)
            
            # Compare the current 'f' fingerprint with the reference fingerprint
            similarity_score_reference = compare_prints(f_path, reference_f_path)
            # Classify as False Positive if similarity score is 1, True Negative if it's 0
            if similarity_score_reference == 1:
                false_positive_count += 1
            else:
                true_negative_count += 1
            
            # Compare the current 'f' fingerprint with its corresponding 's' fingerprint
            similarity_score_corresponding = compare_prints(f_path, s_path)
            # Classify as True Positive if similarity score is 1, False Negative if it's 0
            if similarity_score_corresponding == 1:
                true_positive_count += 1
            else:
                false_negative_count += 1
    
    # Return counts for each category
    return true_positive_count, true_negative_count, false_positive_count, false_negative_count



# Example usage
if __name__ == "__main__":
    directory_1 = '/home/student/Downloads/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_4/'
    directory_2 = '/home/student/Downloads/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_5/'
    stats = main(directory_1, directory_2)
    print(stats)
