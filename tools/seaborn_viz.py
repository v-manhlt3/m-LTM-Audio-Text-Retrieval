
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
noise = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# text to audio
t2a_r1_ntxent = [34.21, 31.34,29.65,26.68,25.14,20.58]
t2a_r1_triplet = [29.36,23.01,3.21,0.1,0.2,0.1]
t2a_r1_iot = [38.76,35.92,34.9,33.64,30.57,27.73]

t2a_r5_ntxent = [69.11,67.73,64.61,62.98,59.95,53.96]
t2a_r5_triplet = [65.3,54.98,13.47,1.19,0.54,0.52]
t2a_r5_iot = [73.16,72.28,70.8,69.23,65.8,62.61]

t2a_r10_ntxent = [83.67,81.27,79.14,78.18,75.59,70.72]
t2a_r10_triplet = [71.05,69.98,22.77,2.75,1.67,1.06]
t2a_r10_iot = [85.47,84.11,83.01,82.27,79.39,76.17]

# audio to text
a2t_r1_ntxent = [43.39,40.12,36.78,34.69,32.18,27.37]
a2t_r1_triplet = [37.04,28.52,5.85,1.25,1.14,0.1]
a2t_r1_iot = [47.85,47.12,45.03,42.63,39.91,35.42]

a2t_r5_ntxent = [72.74,70.84,70.11,66.66,63,58.72]
a2t_r5_triplet = [71.05,58.09,19.64,5.43,3.76,0.52]
a2t_r5_iot = [80.66,79.2,78.36,73.35,71.47,68.65]

a2t_r10_ntxent = [85.33,82.54,81.5,78.99,78.05,75.21]
a2t_r10_triplet = [81.92,70.11,31.13,9.4,6.47,1.46]
a2t_r10_iot = [89.75,88.19,87.77,86.1,83.17,80.56]

data = {'noise': noise, 'NTXent': a2t_r1_ntxent, 
        'Triplet': a2t_r1_triplet, 'IOT': a2t_r1_iot}
data = pd.DataFrame(data)
data = pd.melt(data, ['noise'])
data.columns = ["Noise", "Methods", 'R@1']
# data.rename({'variable':'R@1'}, axis=1, in)

# print(pd.melt(data, ['noise']))
plt.rcParams.update({'font.size': 16})

plot = sns.lineplot(x='Noise', y="R@1", hue="Methods", data=data,
             markers=True, style='Methods', dashes=False)
sns.move_legend(plot, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
plot.set_xlabel("Noise", fontsize=14)
plot.set_ylabel("R@1", fontsize=20)
# plt.legend(loc='low left')
plt.savefig("a2t-R1-ranking.pdf")

