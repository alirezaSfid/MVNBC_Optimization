import ROfeas
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random as r
import gurobipy as gp
from gurobipy import GRB
import csv

import pandas


datadir="Exp_2/Data/"
kerneldir="Exp_2/Uncertainty_set/kernel/"
nndir="Exp_2/Uncertainty_set/nn/"
nnvalsdir="Exp_2/Solutions/t-x/nn/"
kernelvalsdir="Exp_2/Solutions/t-x/kernel/"



for A in [1, 2, 3]:

        for C in range(1,11,1):
            print("A C", A, C)
            
            
            fileName = datadir+"test-"+str(A)+"-"+str(C)+".txt"
            X_test = np.genfromtxt(fileName, delimiter=',')
            fileName = datadir+"train-"+str(A)+"-"+str(C)+".txt"
            X_train = np.genfromtxt(fileName, delimiter=',')
            
            N=X_train.shape[1]
            
            RHS = 50 * N            
            
            for quantil in [0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
                
                #solve kernel
                #Q,theta,alphas,SV
                Q = np.array(pandas.read_csv(kerneldir+"q-"+str(A)+"-"+str(C)+"-"+("%.2f" % quantil)+".txt", header=None))
                theta = np.array(pandas.read_csv(kerneldir+"theta-"+str(A)+"-"+str(C)+"-"+("%.2f" % quantil)+".txt", header=None))[0]
                SV = np.array(pandas.read_csv(kerneldir+"sv-"+str(A)+"-"+str(C)+"-"+("%.2f" % quantil)+".txt", header=None))[0]
                alphas = np.array(pandas.read_csv(kerneldir+"alphas-"+str(A)+"-"+str(C)+"-"+("%.2f" % quantil)+".txt", header=None))[0]
                
                start = time.time()
                objKern, x_kern = ROfeas.solveKernel(N,X_train,Q,theta,alphas,SV,RHS)
                end = time.time()
                t_k = end-start
                filename = kernelvalsdir+"vals-"+str(A)+"-"+str(C)+"-"+str(quantil)+".txt"
                # Open the file in write mode ('w') and write t and x to it
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['t', 'x'])  # write the header
                    writer.writerow([t_k, x_kern])  # write the data
                    
                
            #solve neural network
            
            for quantil in [0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
                E = 3
                L = 3
                fileName = nndir+"c-"+str(A)+"-"+str(C)+"-"+str(E)+".txt"
                c0 = np.genfromtxt(fileName, delimiter=',')
                
                listW = []
                dimLayers = []
                
                for F in range(0,L,1):
                    fileName = nndir+"W-"+str(A)+"-"+str(C)+"-"+str(E)+"-"+str(F+1)+".txt"
                    listW.append(np.genfromtxt(fileName, delimiter=','))
                    dimLayers.append(listW[F].shape[0])
                    
                N=listW[0].shape[1]
                
                
                maxScenEntry = max(np.amax(X_train),np.amax(X_test))
                maxEntry = max(np.amax(X_train),np.amax(X_test))
                M=[]
                for i in range(0,L,1):
                    rowSums = np.sum(np.absolute(listW[i]),axis=1)
                    M.append(maxEntry*np.amax(rowSums))
                    maxEntry = maxEntry*np.amax(rowSums)
                
                
                    
                R,sigmas,lb,ub = ROfeas.getRadiiDataPoints(L,c0,X_train,listW, 0, quantil)
                    
                    
                start = time.time()
                obj, x = ROfeas.solveRobustSelection(N,L,dimLayers,c0, R, listW, M, lb, ub, sigmas,RHS)
                end = time.time()
                t = end-start
                filename = nnvalsdir+"vals-"+str(A)+"-"+str(C)+"-"+str(quantil)+".txt"
                # Open the file in write mode ('w') and write t and x to it
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['t', 'x'])
                    writer.writerow([t, x])
                