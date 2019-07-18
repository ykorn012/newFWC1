#======================================================================================================
# !/usr/bin/env python
# title          : FWC_P1_Simulator.py
# description    : Semiconductor Fab Wide Control using FDC, VM, R2R, L2L
# author         : Youngil Jung
# date           : 2018-07-31
# version        : v0.8
# usage          : python GlobalW2W_FWC_Run.py
# notes          : Reference Paper "An Approach for Factory-Wide Control Utilizing Virtual Metrology"
# python_version : v3.5.3
#======================================================================================================
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics

class VM_Process1_시뮬레이터:
    metric = 0

    def __init__(self, A, d, C, seed):
        self.pls = PLSRegression(n_components=6, scale=False, max_iter=50000, copy=True)
        np.random.seed(seed)
        self.A = A
        self.d = d
        self.C = C

    def setParemeter(self, A, d, C):
        self.A = A
        self.d = d
        self.C = C

    def sampling_up(self):
        u1 = np.random.normal(0.4, np.sqrt(0.2))
        u2 = np.random.normal(0.6, np.sqrt(0.2))
        u = np.array([u1, u2])
        return u

    def sampling_vp(self):
        v1 = np.random.normal(1, np.sqrt(0.2))
        v2 = 2 * v1
        v3 = np.random.uniform(0.2, 1.2)
        v4 = 3 * v3
        v5 = np.random.uniform(0, 0.4)
        v6 = np.random.normal(-0.6, np.sqrt(0.2))

        v = np.array([v1, v2, v3, v4, v5, v6])
        return v

    def sampling_ep(self):
        e1 = np.random.normal(0, np.sqrt(0.1))
        e2 = np.random.normal(0, np.sqrt(0.2))
        e = np.array([e1, e2])
        return e

    def sampling(self, k, uk=np.array([0, 0]), vp=np.array([0, 0, 0, 0, 0, 0]), ep=np.array([0, 0]), isInit=True):
        u1 = uk[0]
        u2 = uk[1]
        u = uk

        v1 = vp[0]
        v2 = vp[1]
        v3 = vp[2]
        v4 = vp[3]
        v5 = vp[4]
        v6 = vp[5]

        v = vp
        e = ep

        if isInit == True:
            k1 = k % 100
            k2 = k % 200
            e = np.array([0, 0])   #DoE는 Sampling Actual이기 때문에 e가 없다.
        else:
            k1 = k % 100  # n = 100 일 때 #1 entity maintenance event
            k2 = k % 200  # n = 200 일 때 #1 entity maintenance event
        eta_k = np.array([[k1], [k2]])

        psi = np.array([u1, u2, v1, v2, v3, v4, v5, v6, k1, k2])
        y = u.dot(self.A) + v.dot(self.C) + np.sum(eta_k * self.d, axis=0) + e
        rows = np.r_[psi, y]
        idx_end = len(rows)
        idx_start = idx_end - 2
        return idx_start, idx_end, rows

    def pls_update(self, V, Y):
        self.pls.fit(V, Y)
        return self.pls

    def setDoE_Mean(self, DoE_Mean):
        self.DoE_Mean = DoE_Mean

    def getDoE_Mean(self):
        return self.DoE_Mean

    def setPlsWindow(self, PlsWindow):
        self.PlsWindow = PlsWindow

    def getPlsWindow(self):
        return self.PlsWindow

    def DoE_Run(self, lamda_PLS, Z, M):  ##12, 10
        N = Z * M
        DoE_Queue = []

        for k in range(1, N + 1):      # range(101) = [1, 2, ..., 120])
            idx_start, idx_end, result = self.sampling(k, self.sampling_up(), self.sampling_vp(), self.sampling_ep(), True)
            DoE_Queue.append(result)

        initplsWindow = DoE_Queue.copy()
        npPlsWindow = np.array(initplsWindow)

        plsWindow = []

        for z in np.arange(0, Z):
            npPlsWindow[z * M:(z + 1) * M - 1, 0:idx_start] = lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:idx_start]
            npPlsWindow[z * M:(z + 1) * M - 1, idx_start:idx_end] = lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, idx_start:idx_end])

        for i in range(len(npPlsWindow)):
            plsWindow.append(npPlsWindow[i])

        npDoE_Queue = np.array(plsWindow)
        DoE_Mean = np.mean(npDoE_Queue, axis=0)

        plsModelData = npDoE_Queue - DoE_Mean
        V0 = plsModelData[:, 0:idx_start]
        Y0 = plsModelData[:, idx_start:idx_end]

        pls = self.pls_update(V0, Y0)

        #print('Init VM Coefficients: \n', pls.coef_)

        y_prd = pls.predict(V0) + DoE_Mean[idx_start:idx_end]
        y_act = npDoE_Queue[:, idx_start:idx_end]

        print("Init DoE VM Mean squared error: %.4f" % metrics.mean_squared_error(y_act[:,0:1], y_prd[:,0:1]))
        print("Init DoE VM r2 score: %.4f" % metrics.r2_score(y_act[:,0:1], y_prd[:,0:1]))

        self.setDoE_Mean(DoE_Mean)
        self.setPlsWindow(plsWindow)
        # self.plt_show1(N, y_act[:,0:1], y_prd[:,0:1])

    def VM_Run(self, lamda_PLS, Z, M):
        N = Z * M

        ## V0, Y0 Mean Center
        DoE_Mean = self.getDoE_Mean()
        idx_end = len(DoE_Mean)
        idx_start = idx_end - 2
        meanVz = DoE_Mean[0:idx_start]
        meanYz = DoE_Mean[idx_start:idx_end]

        M_Queue = []
        ez_Queue = []
        mape_Queue = []
        ez_Queue.append([0, 0])
        y_act = []
        y_prd = []
        VM_Output = []
        ACT_Output = []

        plsWindow = self.getPlsWindow()

        for z in np.arange(0, Z):
            for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
                idx_start, idx_end, result = self.sampling(k, self.sampling_up(), self.sampling_vp(), self.sampling_ep(), False)
                psiK = result[0:idx_start]
                psiKStar = psiK - meanVz
                y_predK = self.pls.predict(psiKStar.reshape(1, idx_start)) + meanYz
                rows = np.r_[result, y_predK.reshape(2, )]
                M_Queue.append(rows)

                y_prd.append(rows[idx_end:idx_end + 2])
                y_act.append(rows[idx_start:idx_end])

            del plsWindow[0:M]

            # for i in range(M):  # VM_Output 구한다. lamda_pls 가중치를 반영하지 않는다.
            #     if i == M - 1:
            #         temp = npM_Queue[i:i + 1, idx_start:idx_end]
            #     else:
            #         temp = npM_Queue[i:i + 1, idx_end:idx_end + 2]
            #     VM_Output.append(np.array([temp[0, 0], temp[0, 1]]))

            ez = M_Queue[M - 1][idx_start:idx_end] - M_Queue[M - 1][idx_end:idx_end + 2]
            ez_Queue.append(ez)

            if z == 0:
                ez = np.array([0, 0])
            npVM_Queue = np.array(M_Queue)
            npACT_Queue = np.array(M_Queue)

            npVM_Queue[0:M - 1, 0:idx_start] = lamda_PLS * npVM_Queue[0:M - 1, 0:idx_start]
            npVM_Queue[0:M - 1, idx_start:idx_end] = lamda_PLS * (npVM_Queue[0:M - 1, idx_end:idx_end + 2] + 0.5 * ez) # + 0.5 * ez
            npVM_Queue = npVM_Queue[:, 0:idx_end]  ##idx_start ~ end 까지 VM 값 정리

            npACT_Queue[0:M - 1, 0:idx_start] = lamda_PLS * npACT_Queue[0:M - 1, 0:idx_start]
            npACT_Queue[0:M - 1, idx_start:idx_end] = lamda_PLS * npACT_Queue[0:M - 1, idx_start:idx_end]
            npACT_Queue = npACT_Queue[:, 0:idx_end]  ##idx_start ~ end 까지 VM 값 정리

            for i in range(M):  #VM_Output 구한다. lamda_pls 가중치를 반영하여 다음 계산시 편리하게 한다.
                if i == M - 1:
                    temp = npACT_Queue[i:i + 1, idx_start:idx_end]
                else:
                    temp = npVM_Queue[i:i + 1, idx_start:idx_end]
                VM_Output.append(np.array([temp[0, 0], temp[0, 1]]))
                temp = npACT_Queue[i:i + 1, idx_start:idx_end]
                ACT_Output.append(np.array([temp[0, 0], temp[0, 1]]))

            for i in range(M):
                plsWindow.append(npVM_Queue[i])

            M_Mean = np.mean(plsWindow, axis=0)
            meanVz = M_Mean[0:idx_start]
            meanYz = M_Mean[idx_start:idx_end]

            plsModelData = plsWindow - M_Mean
            V = plsModelData[:, 0:idx_start]
            Y = plsModelData[:, idx_start:idx_end]

            self.pls_update(V, Y)

            del M_Queue[0:M]

        y_act = np.array(y_act)
        y_prd = np.array(y_prd)

        self.metric = metrics.explained_variance_score(y_act[:,0:1], y_prd[:,0:1])
        print("VM Mean squared error: %.3f" % metrics.mean_squared_error(y_act[:,0:1], y_prd[:,0:1]))
        print("explained_variance_score: %.3f" % self.metric)
        print("VM r2 score: %.3f" % metrics.r2_score(y_act[:,0:1], y_prd[:,0:1]))
        ez_run = np.array(ez_Queue)

        VM_Output = np.array(VM_Output)
        ACT_Output = np.array(ACT_Output)

        return VM_Output, ACT_Output, ez_run, y_act, y_prd
