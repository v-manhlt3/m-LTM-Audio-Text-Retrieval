
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def draw_noisy_corres():
        noise = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        # text to audio
        t2a_r1_ntxent = [34.21, 31.34,29.65,26.68,25.14,20.58]
        t2a_r1_triplet = [29.36,23.01,3.21,0.1,0.2,0.1]
        t2a_r1_iot = [38.76,35.92,34.9,33.64,30.57,27.73]
        t2a_r1_ot = [36.57,35.51,35.21,32.58,29.28,26.26]

        t2a_r5_ntxent = [69.11,67.73,64.61,62.98,59.95,53.96]
        t2a_r5_triplet = [65.3,54.98,13.47,1.19,0.54,0.52]
        t2a_r5_iot = [73.16,72.28,70.8,69.23,65.8,62.61]
        t2a_r5_ot = [71.66,71.32,70.86,67.75,64.09,59.72]

        t2a_r10_ntxent = [83.67,81.27,79.14,78.18,75.59,70.72]
        t2a_r10_triplet = [71.05,69.98,22.77,2.75,1.67,1.06]
        t2a_r10_iot = [85.47,84.11,83.01,82.27,79.39,76.17]
        t2a_r10_ot = [84.2,84.01,83.49,80.89,78.66,75.03]

        # audio to text
        a2t_r1_ntxent = [43.39,40.12,36.78,34.69,32.18,27.37]
        a2t_r1_triplet = [37.04,28.52,5.85,1.25,1.14,0.1]
        a2t_r1_iot = [47.85,47.12,45.03,42.63,39.91,35.42]
        a2t_r1_ot = [46.49,46.64,42.84,40.31,37.09,34.48]

        a2t_r5_ntxent = [72.74,70.84,70.11,66.66,63,58.72]
        a2t_r5_triplet = [71.05,58.09,19.64,5.43,3.76,0.52]
        a2t_r5_iot = [80.66,79.2,78.36,73.35,71.47,68.65]
        a2t_r5_ot = [78.05,78.68,74.61,71.16,69.27,66.77]

        a2t_r10_ntxent = [85.33,82.54,81.5,78.99,78.05,75.21]
        a2t_r10_triplet = [81.92,70.11,31.13,9.4,6.47,1.46]
        a2t_r10_iot = [89.75,88.19,87.77,86.1,83.17,80.56]
        a2t_r1_ot = [88.4,87.87,85.68,84.57,82.86,79.62]

        data = {'noise': noise, 'NTXent': t2a_r10_ntxent, 
                'Triplet': t2a_r10_triplet, 'POT': t2a_r10_iot, "OT": t2a_r10_ot}

        # data = {'noise': noise, 'NTXent': t2a_r1_ntxent, 
        #         'POT': t2a_r1_iot, "OT": t2a_r1_ot}
        data = pd.DataFrame(data)
        data = pd.melt(data, ['noise'])
        data.columns = ["Noise", "Methods", 'R@10']
        # data.rename({'variable':'R@1'}, axis=1, in)

        # print(pd.melt(data, ['noise']))
        plt.rcParams.update({'font.size': 16})

        plot = sns.lineplot(x='Noise', y="R@10", hue="Methods", data=data, style='Methods', dashes=False, markers=True)
        sns.move_legend(plot, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False)
        plot.set_xlabel("Noise", fontsize=14)
        plot.set_ylabel("R@10", fontsize=20)
        plt.grid()
        # plt.legend(loc='low left')
        plot.figure.savefig("t2a-R10-ranking.png")

def draw_ablation():
        m = [0.9, 0.8, 0.7, 0.6]

        p02_r1_a2t = [47.12,47.12,46.29,46.18]
        p02_r1_t2a = [37.31, 37.16, 36.9, 36.17]
        p02_r1 = [42.215,42.14, 41.595, 41.175]

        p04_r1_a2t = [42.84, 43.46, 42.52, 41.55]
        p04_r1_t2a = [32.44, 33.44, 32.58, 31.23]
        p04_r1 = [37.64, 38.45, 37.55, 36.39]


        p06_r1_a2t = [35.73, 35.88, 36.57, 35.84]
        p06_r1_t2a = [27.83, 27.87, 28.35, 26.14]
        p06_r1 = [31.78, 31.67, 32.46, 30.99]

        data = {'mass': m, 'noise=0.2': p02_r1,'noise=0.4': p04_r1, 'noise=0.6': p06_r1}
        data = pd.DataFrame(data)
        data = pd.melt(data, ['mass'])
        data.columns = ['Mass', "Noise", "R@1"]
        plt.rcParams.update({'font.size': 16})

        plot = sns.lineplot(x='Mass', y="R@1", hue="Noise", data=data, style='Noise', dashes=False, markers=True)
        sns.move_legend(plot, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False)
        plot.set_xlabel("Mass", fontsize=14)
        plot.set_ylabel("Average R@1", fontsize=20)
        plt.xticks(m)
        plt.grid()
        # plt.legend(loc='low left')
        plot.figure.savefig("Aba-mass.png")

if __name__ == "__main__":
        draw_ablation()
