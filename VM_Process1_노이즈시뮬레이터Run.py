#======================================================================================================
# !/usr/bin/env python
# title          : LocalW2W_FWC_Run.py
# description    : Semiconductor Fab Wide Control using FDC, VM, R2R, L2L
# author         : Youngil Jung
# date           : 2019-06-17
# version        : v0.8
# usage          : python LocalW2W_FWC_Run.py
# notes          : Reference Paper "Virtual metrology and feedback control for semiconductor manufacturing"
# python_version : v3.5.3
#======================================================================================================
import numpy as np
from simulator.VM_Process1_노이즈시뮬레이터 import VM_Process1_노이즈시뮬레이터
from simulator.FDC_Graph import FDC_Graph
# from pandas import DataFrame, Series
# import pandas as pd
import os

os.chdir("D:/10. 대학원/04. Source/OnlyVM/02. Local VM/")

A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])    #recipe gain matrix
d_p1 = np.array([[0.1, 0], [0.05, 0]])  #drift matrix
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]])) # FDC variable matrix
SEED = 999999000

# Process 변수와 출력 관련 system gain matrix

def main():
    fdh_graph = FDC_Graph()
    fwc_p1_vm = VM_Process1_노이즈시뮬레이터(A_p1, d_p1, C_p1, SEED)
    fwc_p1_vm.DoE_Run(lamda_PLS=0.1, Z=12, M=10)  #DoE Run
    VM_Output, ACT_Output, ez_run, y_act, y_prd = fwc_p1_vm.VM_Run(lamda_PLS=0.1, Z=40, M=10)

    fdh_graph.plt_show1(400, y_act[:, 0:1], y_prd[:, 0:1])
    fdh_graph.plt_show2(40 + 1, ez_run[:, 0:1], ez_run[:, 1:2])

    p1_mape_Queue = []
    M = 10
    # for z in np.arange(22, 34, 1):
    #     mape = fdh_graph.mean_absolute_percentage_error(z + 1, y_act[((z + 1) * M) - 1][0], y_prd[((z + 1) * M) - 1][0])
    #     p1_mape_Queue.append(mape)

    for z in np.arange(21, 32, 1):
        mape_sum = 0
        for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
            mape = fdh_graph.mean_absolute_percentage_error(k, y_act[k][0], y_prd[k][0])
            mape_sum += mape
        p1_mape_Queue.append(mape_sum/M)

    print('MAPE (%) : ', np.mean(p1_mape_Queue))
    np.savetxt("output/noise_mape_Queue.csv", p1_mape_Queue, delimiter=",", fmt="%.4f")
    np.savetxt("output/abNormal_VM_Output.csv", VM_Output, delimiter=",", fmt="%.4f")

if __name__ == "__main__":
    main()
