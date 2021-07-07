from cv2 import cv2
import numpy as np

from Airlight import Airlight
from BoundCon import BoundCon
from CalTransmission import CalTransmission
from removeHaze import removeHaze

def dehaze(HazeImg):
    # HazeImg = cv2.imread('i2.png')
    windowSze = 15  # Estimate Airlight
    AirlightMethod = 'fast'
    A = Airlight(HazeImg, AirlightMethod, windowSze)
    windowSze = 3   # Calculate Boundary Constraints
    C0 = 20         # Default value = 20 (as recommended in the paper)
    C1 = 300        # Default value = 300 (as recommended in the paper)
    Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)                  #   Computing the Transmission using equation (7) in the paper
    # Refine estimate of transmission
    regularize_lambda = 1       # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    sigma = 0.5
    Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)     # Using contextual information
    HazeCorrectedImg = removeHaze(HazeImg, Transmission, A, 0.85)   # Perform DeHazing
    return HazeCorrectedImg
    
HazeImg = cv2.imread('./9I.jpg')
HazeCorrectedImg = dehaze(HazeImg)
# cv2.imshow('Result', HazeCorrectedImg)
# cv2.waitKey(0)
cv2.imwrite('result.jpg', HazeCorrectedImg)