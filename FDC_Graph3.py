import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt1
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd

class FDC_Graph:

    def plt_show1(self, n, y_act, y_prd):
        plt.figure()
        plt.plot(np.arange(n), y_act, 'rx--', y_prd, 'bx--', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 50))
        plt.xlabel('Run No.')
        plt.ylabel('Actual and Predicted Response (y1)')

    def plt_show2(self, n, y1, y2, Noise):
        plt.figure()
        plt.plot(np.arange(0, n + 1, 1), y1, 'bx-', y2, 'gx--', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 5))
        if Noise == False:
            plt.yticks(np.arange(-10, 10, 0.2))
        else:
            plt.yticks(np.arange(-5.5, 6, 0.5))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel('e(z)')
        plt.tight_layout()

    def plt_show3(self, n, y1, y2):
        plt.figure()
        plt.plot(np.arange(n), y1, 'bx-', y2, 'gx--', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 5))
        plt.yticks(np.arange(-12, 3, 2))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel('e(z)')

    def plt_show4(self, n, y1):
        plt.figure()
        plt.plot(np.arange(n), y1, 'rx-', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 5))
        plt.yticks(np.arange(-1.2, 1.3, 0.2))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel('e(z)')

    def plt_show5(self, ez_run, N, M, dM, S1, Noise):
        df = pd.DataFrame(ez_run, columns=['q1', 'q2'])
        label = []
        for i in np.arange(0, N + 1, 1):
            if i <= S1 * M:
                label.append(0)
            else:
                label.append(1)
        df['label'] = pd.Series(label)
        # df.loc[251]['label']

        xdata = []
        y1data = []
        y2data = []
        ldata = []

        plt.figure()

        for i in np.arange(0, N + 1, 1):
            if i < S1 * M and i % M == 0:
                xdata.append(i)
                y1data.append(df.loc[i]['q1'])
                y2data.append(df.loc[i]['q2'])
                ldata.append(0)
                # line1.set_xdata(xdata)
                # line1.set_ydata(y1data)
                # line2.set_xdata(xdata)
                # line2.set_ydata(y2data)
            if i >= S1 * M and i % dM == 0:
                xdata.append(i)
                y1data.append(df.loc[i]['q1'])
                y2data.append(df.loc[i]['q2'])
                ldata.append(1)
                # line1.set_xdata(xdata)
                # line1.set_ydata(y1data)
                # line2.set_xdata(xdata)
                # line2.set_ydata(y2data)

        # line1.set_xdata(xdata)
        # line1.set_ydata(y1data)
        # line2.set_xdata(xdata)
        # line2.set_ydata(y2data)

        # plt.show()

        df2 = pd.DataFrame(np.array([xdata, y1data, y2data, ldata]))
        df2 = df2.T
        df2.columns = ['no', 'q1', 'q2', 'label']

        num_classes = 2
        # cmap = ListedColormap(['r', 'g', 'b', 'y'])
        cmap = ListedColormap(['b', 'r'])
        norm = BoundaryNorm(range(num_classes + 1), cmap.N)
        points = np.array([df2['no'], df2['q1']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(df2['label'])

        # fig1 = plt.figure()

        plt.gca().add_collection(lc)
        # plt.xlim(df.index.min(), df.index.max())
        # plt.ylim(-1.1, 1.1)
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel('e(z)')
        # plt.xticks(np.arange(0, 410, 50))
        ticks = np.arange(0, N + 1, 50)
        plt.xticks(ticks)
        if Noise == False:
            plt.yticks(np.arange(-1.2, 1.3, 0.2))
        else:
            plt.yticks(np.arange(-5.5, 6, 0.5))

        dic = {50: "50 \n (5 runs)", 100: "100 \n (10 runs)", 150: "150 \n (15 runs)", 200: "200 \n (20 runs)",
               250: "250 \n (30 runs)", 300: "300 \n (40 runs)", 350: "350 \n (50 runs)", 400: "400 \n (60 runs)"}
        labels = [ticks[i] if t not in dic.keys() else dic[t] for i, t in enumerate(ticks)]

        axes = plt.gca()
        # axes.set_xlim(0, N)
        # axes.set_ylim(-1.2, 1.2)
        # line1, = axes.plot(xdata, y1data, 'b', lw=2, ms=5, mew=2, linestyle='--')
        # line2, = axes.plot(xdata, y2data, 'g', lw=2, ms=5, mew=2, linestyle='--')

        i = 0
        for text in axes.get_xticklabels():
            if i >= 3:
                text.set_color("red")
            i = i + 1

        # ax = fig1.add_subplot(111)
        axes.set_xticklabels(labels)
        # axes.set_color_cycle(colors)
        #plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.markersize'] = 5
        plt.rcParams['lines.markeredgewidth'] = 2
        #plt.rcParams['lines.linestyle'] = 'x--'
        plt.tight_layout()
        #plt.rcParams.update({'font.size': 20, 'lines.linewidth': 3, 'lines.markersize': 15})
        plt.show()

    def mean_absolute_percentage_error(self, z, y_act, y_prd):
        #print('z: ', z, 'y_act : ', y_act, 'y_prd : ', y_prd)
        mape = np.mean(np.abs((y_act - y_prd) / y_act)) * 100
        #print('mape : ', mape)
        return mape
