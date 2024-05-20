import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def np_pretty_print(prec):
    np.set_printoptions(suppress=True, precision=prec)

########################
# FUNCTIONS FROM EX_1  #
########################

# conv to inhomogenous
# ph: shape(4 rows, n cols), n = num of points
def Pi(ph):
    return ph[:-1,:]/ph[-1,:]

# conv to homogenous
# ph: shape(3 rows, n cols), n = num of points
def PiInv(p):
    return np.vstack((p, np.ones(p.shape[1])))

########################
# FUNCTIONS FROM EX_2  #
########################

# P: points in 3D, shape(3 rows, n cols), n = num of points
# distCoeffs: list of coeffs
def distort(P, distCoeffs):
    P = Pi(P)
    r = np.linalg.norm(P, axis=0)
    dr = np.zeros(r.shape)
    power = 2
    for k in distCoeffs:
        dr += k * pow(r, power)
        power += 2

    return PiInv(P * (1 + dr))

# Dimensions:
# K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# t = np.array([[0, 0, 4]])
# Q: shape(3 rows, n cols), n = num of points
# distCoeffs: list of coeffs, empty by default
def projectpoints(K, R, t, Q, distCoeffs=[]):
    Rt = np.concatenate((R, t.T), axis=1)
    return Pi(K@distort(Rt@PiInv(Q), distCoeffs))
    # returns Pi(K@Rt@Q) if no distortion

def crossOp(p):
    x = p[0][0]
    y = p[0][1]
    z = p[0][2]
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]])

# Homography estimation
# q1 = H @ q2
# q1, q2: shape(2 rows, n cols), n = num of points
def hest(q1, q2, normalize=False):
    if normalize:
        q1, T1 = normalize2d(q1)
        q2, T2 = normalize2d(q2)
    B = []
    q1 = PiInv(q1)
    q2 = PiInv(q2)
    for i in range(q1.shape[1]):
        q1ix = crossOp(np.expand_dims(q1[:,i], axis=0))
        q2i = np.expand_dims(q2[:,i], axis=0)
        B.append(np.kron(q2i, q1ix))
    B = np.vstack(B)
    U, S, VT = np.linalg.svd(B)
    H_flat = np.expand_dims(VT[-1], axis=0)
    H = H_flat.reshape((3, 3)).T
    if normalize:
        return np.linalg.inv(T1)@H@T2
    else:
        return H

#[1/xs, 0,    -xm/xs]
#[0,    1/ys, -ym/ys]
#[0,    0,     1]
# return, p: shape(2 rows, n cols), n = num of points
def normalize2d(p):
    mean = np.mean(p,axis=1)
    std = np.std(p,axis=1)
    T = np.array([[1/std[0],0,-mean[0]/std[0]],
                  [0,1/std[1],-mean[1]/std[1]],
                  [0,0,1]])
    return Pi(T@PiInv(p)), T

########################
# FUNCTIONS FROM EX_3  #
########################

# q: shape(2 rows, n cols), n = num of points
# q is the same point viewed from different viewpoints in image coords
# P: list of 3x4 proj mtces of length n
# ret: 4x1 homogenious 3D triangulated point
def triangulate(q, P):
    B = []
    for i in range(q.shape[1]):
        B.append(P[i][2] * q[0][i] - P[i][0])
        B.append(P[i][2] * q[1][i] - P[i][1])
    B = np.vstack(B)
    U, S, VT = np.linalg.svd(B)
    return np.expand_dims(VT[-1], axis=0).T

# Dimensions:
# K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# t = np.array([[0, 0, 4]])
# Retuns the 3x4 proj mtx
def getP(K, R, t):
    Rt = np.concatenate((R, t.T), axis=1)
    return K @ Rt

########################
# FUNCTIONS FROM EX_4  #
########################

# Estimates the Projection matrix that projects points Q to q
# Q: shape(3 rows, n cols), n = num of points
# 3d points in inhomogenous coords
# q: shape(3 rows, n cols), n = num of points (same as Q)
# 2d points in homogenous coords
def pest(Q, q):
    B = []
    for i in range(Q.shape[1]):
        qix = crossOp(np.expand_dims(q[:,i]/q[-1,i], axis=0))
        Qi = PiInv(np.expand_dims(Q[:,i], axis=0).T).T
        B.append(np.kron(Qi, qix))
    B = np.vstack(B)
    U, S, VT = np.linalg.svd(B)
    P_flat = np.expand_dims(VT[-1], axis=0)
    return P_flat.reshape((4, 3)).T

########################
# FUNCTIONS FROM EX_6  #
########################

def gaussian(x,sigma):
    return np.exp(-x**2 / (2*sigma**2))

def gausDerivative(x,sigma):
    return -(x/sigma)*gaussian(x,sigma)

def gaussian1DKernel(sigma,size=5):
    x = np.arange(-size*sigma, size*sigma+1)
    gaus = gaussian(x,sigma)
    gausD = gausDerivative(x,sigma)
    gaus /= gaus.sum()
    return np.array([gaus]), np.array([gausD])

def gaussianSmoothing(im, sigma):
    g, gd = gaussian1DKernel(sigma)
    I = cv2.sepFilter2D(im, -1, g, g)
    Ix = cv2.sepFilter2D(im, -1, gd, g)
    Iy = cv2.sepFilter2D(im, -1, g, gd)
    return I, Ix, Iy

########################
# FUNCTIONS FROM EX_9  #
########################

# Samples 'size' number of point pairs from the matches
# return: two 2D point arrays, each of shape(2 rows, size cols)
def sample(matches, kp1, kp2, size):
    indices = np.random.choice(len(matches), size=size, replace=False)
    sample_kp1 = [kp1[matches[i].queryIdx].pt for i in indices]
    sample_kp2 = [kp2[matches[i].trainIdx].pt for i in indices]
    return np.asarray(sample_kp1).T, np.asarray(sample_kp2).T