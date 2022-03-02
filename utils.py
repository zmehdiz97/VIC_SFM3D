import numpy as np
import cv2 
import numpy as np 
import pickle 
import argparse
import os 
from time import time
from tqdm import tqdm
import joblib
import random

def kp2list(kp): 
    """Keypoint objects to list"""

    out = []
    for kp_ in kp: 
        temp = (kp_.pt, kp_.size, kp_.angle, kp_.response, kp_.octave, kp_.class_id)
        out.append(temp)
    return out

def matches2list(matches): 
    """matchees to list"""

    out = []
    for match in matches: 
        temp = (match.queryIdx, match.trainIdx, match.imgIdx, match.distance) 
        out.append(temp)
    return out

# Apply ratio test
def get_good_matches(matches, dist_ratio=0.75):
  good = []
  for m,n in matches:
      if m.distance < dist_ratio*n.distance:
          good.append(m)
  return good

def list2kp(kp): 
    out = []
    for point in kp:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],size=point[1], angle=point[2],
         response=point[3], octave=point[4], class_id=point[5]) 
        out.append(temp)

    return out

def list2matches(matches): 
    out = []
    for match in matches:
        out.append(cv2.DMatch(match[0],match[1],match[2],match[3])) 
    return out

def rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''
    RMS = lambda data_, ref_: np.sqrt( np.mean( np.sum(np.power(data_ - ref_, 2), axis=0) ) )

    data_mean = data.mean(axis=1, keepdims=True)
    ref_mean = ref.mean(axis=1, keepdims=True)
    
    Q_data = data - data_mean
    Q_ref = ref - ref_mean
    H = Q_data @ Q_ref.T

    U, S, V = np.linalg.svd(H)

    R = V.T @ U.T
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = V.T @ U.T

    T = ref_mean - R @ data_mean
    data_aligned = R @ data + T
    rms = RMS(data_aligned, ref)
    return R, T, rms


def ransac_rigid_transform(data, ref,iters=None):
    '''
    Computes the Ransac least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''
    assert data.shape == ref.shape
    N = data.shape[1]

    R, T, rms = rigid_transform(data, ref)
    best_R, best_T, best_rms = R, T, rms
    for i in range(iters):
        n_pts = random.randint(N//4,N)
        mask = np.random.randint(0,N,size=(n_pts,))
        R, T, rms = rigid_transform(data[:,mask], ref[:,mask])
        if rms < best_rms:
            best_R, best_T, best_rms = R, T, rms
    
    return best_R, best_T, best_rms