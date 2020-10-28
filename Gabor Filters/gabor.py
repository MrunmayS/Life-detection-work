import cv2
import numpy as np
import matplotlib.pyplot as plt 
import math

def filteredimages(filters, img):
    feature_imgs = []
    for filter in filters:
        kern, params = filter
        fimg = cv2.filter2D(img, cv2.CV_8UC3,kern)
        feature_imgs.append(fimg)
    return feature_imgs

def selection(feature_imgs, threshold, img, img_nums):
    energy_list = []
    cnt = 0
    height, width = img.shape
    for feature_img in feature_imgs:
        current_energy = 0.0
        for x in range(height):
            for y in range(width):
                current_energy += pow(np.abs(feature_img[x][y]),2)
        energy_list.append((current_energy, cnt))
        cnt += 1
    E = 0.0
    for E_i in energy_list:
        E += E_i[0]
    list_sort = sorted(energy_list, key = lambda energy: energy[0], reverse=True)

    temp_sum = 0.0
    Rsquared = 0.0
    added = 0
    output_feature_imgs = []
    while ((Rsquared <threshold) and (added < img_nums)):
        temp_sum += list_sort[added][0]
        Rsquared = (temp_sum/E)
        output_feature_imgs.append(feature_imgs[list_sort[added][1]])
        added += 1
    return output_feature_imgs

def build_filters(lambdas, ksize, gamma, sig, psi):
    filters = []
    thetas = []

    thetas.extend([0, 45, 90, 135])
    thetasInRadians = [np.deg2rad(x) for x in thetas]

    for lamb in lambdas:
        for theta in thetasInRadians:
            params = {'ksize': (ksize,ksize), 'sigma': sig, 'theta': theta, 'lambd': lamb, 'gamma':gamma, 'psi':psi, 'ktype':cv2.CV_64F}
            kern = cv2.getGaborKernel(**params)
            kern /= 1.5 * kern.sum()
            filters.append((kern, params))
    return filters

def lamda_val(img):
    height, width = img.shape

    max = (width/4) * math.sqrt(2)
    min = 4 * math.sqrt(2)
    temp = min
    radial_freq = []

    while(temp < max):
        radial_freq.append(temp)
        temp = temp * 2
    
    radial_freq.append(max)
    lambdavals = []
    for freq in radial_freq:
        lambdavals.append(width/freq)
    return lambdavals

def nonLinearTransducer(img, gaborImages, L, sigmaWeight, filters):

    alpha_ = 0.25
    featureimages = []
    count = 0
    for gaborImage in gaborImages:

        avgPerRow = np.average(gaborImage, axis=0)
        avg = np.average(avgPerRow, axis=0)
        gaborImage = gaborImage.astype(float) - avg
        gaborImage = cv2.normalize(gaborImage, gaborImage, alpha=-8, beta=8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        height, width = gaborImage.shape
        copy = np.zeros(img.shape)
        for row in range(height):
            for col in range(width):
                copy[row][col] = math.fabs(math.tanh(alpha_ * (gaborImage[row][col])))

        copy, destroyImage = applyGaussian(copy, L, sigmaWeight, filters[count])
        if(not destroyImage):
            featureimages.append(copy)
        count += 1

    return featureimages

def applyGaussian(gaborImage, L, sigmaWeight, filter):

    height, N_c = gaborImage.shape

    nparr = np.array(filter[0])
    u_0 = nparr.mean(axis=0)
    u_0 = u_0.mean(axis=0)

    destroyImage = False
    sig = 1
    if (u_0 < 0.000001):
        print 'div by zero occured for calculation:'
        print "sigma = sigma_weight * (N_c/u_0), sigma will be set to zero"
        print "removing potential feature image!"
        destroyImage = True
    else:
        sig = sigmaWeight * (N_c / u_0)

    return cv2.GaussianBlur(gaborImage, (L, L), sig), destroyImage

def removeFeatureImagesWithSmallVariance(featureImages, threshold):
    toReturn =[]
    for image in featureImages:
        if(np.var(image) > threshold):
            toReturn.append(image)
    for i in range(len(toReturn)):
        plt.imshow(i)


img = cv2.imread('E:\Life Detection\Gabor Filters\img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ksize = 5
sigma = 3
gamma = 0.5
psi = 0
R_thresh = 0.95
lamdas = lamda_val(img)
filters = build_filters(lamdas, ksize, gamma, sigma,psi)
filteredimgs = filteredimages(filters, img)
filteredimgs = selection(filteredimgs,R_thresh, img, 1)

featureimgs = nonLinearTransducer(img,filteredimgs, 10, 0.5,filters)
featureimgs = removeFeatureImagesWithSmallVariance(filteredimgs, 0.0001)

