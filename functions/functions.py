import numpy as np
import cv2

def CalculateHistogram(img):
    M, N = img.shape[:2]
    histo = np.zeros(256, dtype=int)

    for i in range(M):
        for j in range(N):
            histo[int(img[i,j])] += 1
    return histo

def CalculateHistogramHSV(img):
    histoH = np.zeros(180, dtype=int)
    histoS = np.zeros(256, dtype=int)
    histoV = np.zeros(256, dtype=int)


    H, S, V = cv2.split(img.astype(np.uint8))
    M, N = S.shape[:2]
    for i in range(M):
        for j in range(N):
            h = min(int(H[i, j]), 179)
            histoH[h] += 1
            histoS[int(S[i, j])] += 1
            histoV[int(V[i, j])] += 1

    return histoH, histoS, histoV

def CDFCalc(histo):
    cdf = np.zeros(len(histo), dtype=int)
    i = 1
    cdf[0] = histo[0]
    for i in range(len(histo)):
        cdf[i] = cdf[i-1] + histo[i]
    return cdf

# Works only on uint8
def EqualiseImg(img):
    histo = CalculateHistogram(img)
    cdf = CDFCalc(histo)
    M, N = img.shape[:2]
    cdf_min = cdf[0]

    equalisation = np.round(((cdf - cdf_min) / ((M*N) - cdf_min)) * (256-1))
    eqImg = equalisation[img.astype(np.uint8)]
    return eqImg

def EqualizeImgHSV(img):
    histoH, histoS, histoV = CalculateHistogramHSV(img)
    cdf = CDFCalc(histoV)
    M, N = img.shape[:2]
    cdf_min = cdf[0]

    equalisation = np.round(((cdf - cdf_min) / ((M * N) - cdf_min)) * (256 - 1))
    eqImg = equalisation[img.astype(np.uint8)]
    return eqImg