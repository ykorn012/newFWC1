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

class VM_Process2_노이즈시뮬레이터:
    metric = 0

    def __init__(self, A, d, C, F, p_lambda, p_VM, p_ACT, seed):
        self.pls = PLSRegression(n_components=6, scale=False, max_iter=50000, copy=True)
        np.random.seed(seed)
        self.A = A
        self.d = d
        self.C = C
        self.F = F
        self.p_lambda = p_lambda
        self.p_VM = p_VM
        self.p_ACT = p_ACT
        self.real_ACT = []  # Process-1을 반영하는 실제 Actual 값

    def sampling_up(self):
        # u1 = np.random.normal(0.4, np.sqrt(0.2))
        # u2 = np.random.normal(0.6, np.sqrt(0.2))
        u1 = np.random.normal(0.2, np.sqrt(0.1))
        u2 = np.random.normal(0.1, np.sqrt(0.05))
        u = np.array([u1, u2])
        return u

    def sampling_vp(self):
        v1 = np.random.normal(-0.4, np.sqrt(0.2))
        v2 = 2 * v1
        v3 = np.random.uniform(0.2, 0.6)
        v4 = 3 * v3
        v5 = np.random.uniform(0, 0.4)

        v = np.array([v1, v2, v3, v4, v5])
        return v

    def sampling_ep(self):
        e1 = np.random.normal(0, np.sqrt(0.05))
        e2 = np.random.normal(0, np.sqrt(0.1))
        e = np.array([e1, e2])
        return e

    def sampling(self, k, uk=np.array([0, 0]), vp=np.array([0, 0, 0, 0, 0]), ep=np.array([0, 0]), p_VM=np.array([0, 0]), p_ACT=np.array([0, 0]), isInit=True):
        u1 = uk[0]
        u2 = uk[1]
        u = uk

        v1 = vp[0]
        v2 = vp[1]
        v3 = vp[2]
        v4 = vp[3]
        v5 = vp[4]

        v = vp
        e = ep

        k1 = k % 150
        k2 = k
        eta_k = np.array([[k1], [k2]])

        if isInit == True:
            e = np.array([0, 0])   #DoE는 Sampling Actual이기 때문에 e가 없다.
            fp = p_ACT  # DoE에서는 Process-1 Act 값 사용
        else:
            fp = p_VM   # VM에서는 Process-1 VM 값 사용

        A = self.A
        C = self.C
        d = self.d

        if isInit == False and fp is not None:
            if k >= 151:
                # u1 = np.random.normal(1.6, np.sqrt(0.2))
                # u2 = np.random.normal(1.4, np.sqrt(0.2))
                # u = np.array([u1, u2])

                v1 = np.random.normal(0.3, np.sqrt(0.1))
                v2 = 2 * v1
                v3 = np.random.uniform(0.6, 0.9)
                v4 = 3 * v3
                v5 = np.random.uniform(0, 0.4)

                r = 3
                v1 = r * v1
                v2 = r * v2
                v3 = r * v3
                v4 = r * v4
                v5 = r * v5

                v = np.array([v1, v2, v3, v4, v5])
                k1 = 1 * k1
                k2 = 1 * k2
                eta_k = np.array([[k1], [k2]])
                A = 0.2 * A   #0.2
                #A_chg = self.A
                # C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))
                #C = np.transpose(np.array([[0.7, 0.8, 0.5, 0.2, 0.3, 0.2], [0.2, 0.5, 0.8, 0.4, 0.2, 0.2]]))
                C = 0.2 * C    #0.2
                #C_chg = np.transpose(np.array([[0, 0.25, 0.01, 0, 0.05, 0], [0.04, 0, 0.01, 0.1, 0, 0]]))
                #C_chg = np.transpose(np.array([[0.7, 0.8, 0.5, 0.2, 0.3, 0.2], [0.085, 0, 0.025, 0.2, 0, 0]]))
                #d = np.array([[0.14, 0], [0.07, 0]])
                #d_chg = self.d
                #d = 1 * d
                # e1 = np.random.normal(-1, np.sqrt(0.1))
                # e2 = np.random.normal(-1, np.sqrt(0.1))
                # e = np.array([e1, e2])

        psi = np.array([u1, u2, v1, v2, v3, v4, v5, k1, k2])

        if fp is not None:   # Process-1의 입력값이 있다면
            # 이건 왜 해놓은지 모르겠다.. R2R에서 결과를 도출하기 위해서인지, y를 Paremeter로 학습목적인지.. 추후 검증필요
            if k % 10 == 0:
                f = p_ACT
            else:
                f = p_VM
            if isInit == True:
                f = p_ACT
            psi = np.r_[psi, f]
            # VM이든 DoE든 계산한다.
            y = u.dot(self.A) + v.dot(self.C) + np.sum(eta_k * self.d, axis=0) + f.dot(self.F) + e
            if isInit == False: #VM이 아닌 실제 ACT값을 별도로 계산한다.
                #print('f.dot(self.F) : ', f.dot(self.F), 'p_ACT.dot(self.F) : ', p_ACT.dot(self.F))
                temp = u.dot(A) + v.dot(C) + np.sum(eta_k * d, axis=0) + p_ACT.dot(self.F) + e
                self.real_ACT.append(np.array([temp[0], temp[1]]))
        else: # Process-1의 입력값이 없고, 향후 Process-2 VM만 하고 싶을 때
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

    def DoE_Run(self, lamda_PLS, Z, M, f):
        N = Z * M
        DoE_Queue = []

        for k in range(1, N + 1):      # range(101) = [1, 2, ..., 120])
            if f is not None:
                fp = f[k - 1, 0:2]
            else:
                fp = None
            idx_start, idx_end, result = self.sampling(k, self.sampling_up(), self.sampling_vp(), self.sampling_ep(), None, fp, True)
            DoE_Queue.append(result)

        initplsWindow = DoE_Queue.copy()
        npPlsWindow = np.array(initplsWindow)

        plsWindow = []

        # Process-1의 lamda_PLS는 이미 반영되어서 넘어오기 때문에, 중복되어 lamda_PLS를 반영할 필요가 없다.
        for z in np.arange(0, Z):
            if f is not None:
                npPlsWindow[z * M:(z + 1) * M - 1, 0:idx_start - 2] = lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:idx_start - 2]
                npPlsWindow[z * M:(z + 1) * M - 1, idx_start - 2:idx_start] = self.p_lambda * npPlsWindow[z * M:(z + 1) * M - 1, idx_start - 2:idx_start]
                npPlsWindow[z * M:(z + 1) * M - 1, idx_start:idx_end] = lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, idx_start:idx_end])
            else:
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

        y_prd = pls.predict(V0) + DoE_Mean[idx_start:idx_end]
        y_act = npDoE_Queue[:, idx_start:idx_end]

        print("Init DoE VM Mean squared error: %.4f" % metrics.mean_squared_error(y_act[:,1:2], y_prd[:,1:2]))
        print("Init DoE VM r2 score: %.4f" % metrics.r2_score(y_act[:,1:2], y_prd[:,1:2]))
        #print("pls : ", pls.coef_)

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
                if  self.p_VM[k-1] is not None:
                    idx_start, idx_end, result = self.sampling(k, self.sampling_up(), self.sampling_vp(), self.sampling_ep(), self.p_VM[k-1], self.p_ACT[k-1], False)
                else:
                    idx_start, idx_end, result = self.sampling(k, self.sampling_up(), self.sampling_vp(), self.sampling_ep(), None, None, False)
                psiK = result[0:idx_start]
                psiKStar = psiK - meanVz
                y_predK = self.pls.predict(psiKStar.reshape(1, idx_start)) + meanYz
                rows = np.r_[result, y_predK.reshape(2, )]
                M_Queue.append(rows)

                y_prd.append(rows[idx_end:idx_end + 2])
                y_act.append(rows[idx_start:idx_end])

            del plsWindow[0:M]

            ez = M_Queue[M - 1][idx_start:idx_end] - M_Queue[M - 1][idx_end:idx_end + 2]
            ez_Queue.append(ez)

            if z == 0:
                ez = np.array([0, 0])
            npVM_Queue = np.array(M_Queue)
            npACT_Queue = np.array(M_Queue)

            # for i in range(M):  # VM_Output 구한다. lamda_pls 가중치를 반영하지 않는다.
            #     if i == M - 1:
            #         temp = npM_Queue[i:i + 1, idx_start:idx_end]
            #     else:
            #         temp = npM_Queue[i:i + 1, idx_end:idx_end + 2]
            #     VM_Output.append(np.array([temp[0, 0], temp[0, 1]]))

            # Process-1의 lamda_PLS는 이미 반영되어서 넘어오기 때문에, 중복되어 lamda_PLS를 반영할 필요가 없다.
            if  self.p_VM[z-1] is not None:
                npVM_Queue[0:M - 1, 0:idx_start - 2] = lamda_PLS * npVM_Queue[0:M - 1, 0:idx_start - 2]
                npVM_Queue[0:M - 1, idx_start - 2:idx_start] = self.p_lambda * npVM_Queue[0:M - 1, idx_start - 2:idx_start]
                npVM_Queue[0:M - 1, idx_start:idx_end] = lamda_PLS * (npVM_Queue[0:M - 1, idx_end:idx_end + 2] + 0.5 * ez) # + 0.5 * ez
                npVM_Queue = npVM_Queue[:, 0:idx_end]

                npACT_Queue[0:M - 1, 0:idx_start - 2] = lamda_PLS * npACT_Queue[0:M - 1, 0:idx_start - 2]
                npACT_Queue[0:M - 1, idx_start - 2:idx_start] = lamda_PLS * npACT_Queue[0:M - 1, idx_start - 2:idx_start]
                npACT_Queue[0:M - 1, idx_start:idx_end] = lamda_PLS * npACT_Queue[0:M - 1, idx_start:idx_end]
                npACT_Queue = npACT_Queue[:, 0:idx_end]  ##idx_start ~ end 까지 VM 값 정리
            else:
                npVM_Queue[0:M - 1, 0:idx_start] = lamda_PLS * npVM_Queue[0:M - 1, 0:idx_start]
                npVM_Queue[0:M - 1, idx_start:idx_end] = lamda_PLS * (
                npVM_Queue[0:M - 1, idx_end:idx_end + 2] + 0.5 * ez)  # + 0.5 * ez
                npVM_Queue = npVM_Queue[:, 0:idx_end]

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

        #y_act = np.array(y_act)

        y_act = np.array(self.real_ACT)
        y_prd = np.array(y_prd)

        self.metric = metrics.explained_variance_score(y_act[:,1:2], y_prd[:,1:2])
        print("VM Mean squared error: %.4f" % metrics.mean_squared_error(y_act[:,1:2], y_prd[:,1:2]))
        print("explained_variance_score: %.4f" % self.metric)
        print("VM r2 score: %.4f" % metrics.r2_score(y_act[:,1:2], y_prd[:,1:2]))
        #print("pls : ", self.pls.coef_)
        ez_run = np.array(ez_Queue)

        VM_Output = np.array(VM_Output)
        ACT_Output = np.array(ACT_Output)

        return VM_Output, ACT_Output, ez_run, y_act, y_prd
