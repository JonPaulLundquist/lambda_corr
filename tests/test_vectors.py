#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Fri Oct 10 23:27:40 2025

Even with an asymptotic efficiency of 81% the Lambda correlation performs well
against many different correlation measures for samples as small as N = 15.
Tested sample sets are from here:
https://scispace.com/pdf/comparing-the-effectiveness-of-rank-correlation-statistics-3994fjbw8y.pdf?utm_source=chatgpt.com

@author: Jon Paul Lundquist
"""
from lambda_corr import lambda_corr
import numpy as np

#Small samples with lots of noise
A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=int)  # Natural
B = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=int)  # Inverse
C = np.array([1, 2, 3, 4, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5], dtype=int)  # Floor effect
D = np.array([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 13, 14, 15], dtype=int)  # Ceiling effect
E = np.array([1, 2, 3, 4, 11, 10, 9, 8, 7, 6, 5, 12, 13, 14, 15], dtype=int)  # Bipolarity/A
F = np.array([15, 14, 13, 12, 11, 6, 7, 8, 9, 10, 5, 4, 3, 2, 1], dtype=int)  # Bipolarity/D
G = np.array([8, 7, 6, 5, 4, 3, 2, 1, 9, 10, 11, 12, 13, 14, 15], dtype=int)  # U-shaped/LH
H = np.array([15, 14, 13, 12, 11, 10, 9, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)  # U-shaped/HL
I = np.array([1, 2, 3, 4, 5, 6, 7, 8, 15, 14, 13, 12, 11, 10, 9], dtype=int)  # Inverted U/LH
J = np.array([9, 10, 11, 12, 13, 14, 15, 8, 7, 6, 5, 4, 3, 2, 1], dtype=int)  # Inverted U/HL
K = np.array([9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)  # Bilinear/A
L = np.array([8, 7, 6, 5, 4, 3, 2, 1, 15, 14, 13, 12, 11, 10, 9], dtype=int)  # Bilinear/D

#Magnitude only
#Result is 4th for unweighted. 6th compared to weighted
#And sign
#Result is 3rd for unweighted. 2nd compared to weighted
Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a = lambda_corr(A,C,pvals=False)
print(Lambda_s)

#Magnitude only
#Result is 2nd for unweighted. 2nd compared to weighted
Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a = lambda_corr(A,D,pvals=False)
print(Lambda_s)

#Magnitude only
#Result is 6th for unweighted. 8th compared to weighted
Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a = lambda_corr(A,E,pvals=False)
print(Lambda_s)

#Magnitude only
#Result is 5th (tied) for unweighted. 9th compared to weighted
Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a = lambda_corr(A,F,pvals=False)
print(Lambda_s)

#Magnitude only
#Result is 10th for unweighted. 8th compared to weighted
#and sign
##Result is 10th for unweighted. 6th compared to weighted
Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a = lambda_corr(A,G,pvals=False)
print(Lambda_s)

#Magnitude only
#Result is 12th for unweighted. 8th compared to weighted
#and sign
#Result is 11th for unweighted. 7th compared to weighted
Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a = lambda_corr(A,H,pvals=False)
print(Lambda_s)

#Magnitude only
#Result is 3rd for unweighted. 2nd (tied) compared to weighted
Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a = lambda_corr(A,I,pvals=False)
print(Lambda_s)

#Magnitude only
#Result is 5th for unweighted. 1st (tied) compared to weighted
Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a = lambda_corr(A,J,pvals=False)
print(Lambda_s)

#Magnitude only
#Result is 14th for unweighted. 11th (last) compared to weighted
#And sign
#Result is 11th for unweighted. 9th compared to weighted
Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a = lambda_corr(A,K,pvals=False)
print(Lambda_s)

#Magnitude only
#Result is 13th for unweighted. 11th (last) compared to weighted
#and sign
#Result is 8th for unweighted. 8th compared to weighted
Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a = lambda_corr(A,L,pvals=False)
print(Lambda_s)

#OVERALL RANKING: 2nd place for unweighted stats and 8th for weighted.
#Take the rank the values of r_x (for each column C to L) by which have the highest correlations 
#(with correct sign), sum those r_x rankings (over C to L), and rank the r_x by total performance.
#Lambda comes in second place for unweighted stats. 
#The median of sample pair-wise rank slopes r_12 comes in first.
#For weighted stats Lambda comes in 8th. r_12 comes in first.
#This is very good given the small sample sizes and the 81% asymptotic efficiency of Lambda.
