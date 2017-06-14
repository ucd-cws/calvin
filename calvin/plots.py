import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.cm as cm
import matplotlib
matplotlib.style.use('ggplot')


def plot_clustered_stacked(dfall, labels=None, title="Water Supply Portfolio",  H="/", **kwargs):

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall :
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  

    h,l = axe.get_legend_handles_labels() 
    for i in range(0, n_df * n_col, n_col): 
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches:
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))    
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    return axe

fp = '../results-2017-06-14T09-57-08Z'
F = pd.read_csv(fp + '/flow.csv', index_col=0, parse_dates=True)
portfolio = pd.read_csv('data/portfolio.csv', index_col = 0)

new_df_urban = pd.DataFrame(index=portfolio.region.unique())
new_df_ag = pd.DataFrame(index=portfolio.region.unique())

for P in portfolio.region.unique():
    for k in portfolio.supplytype.unique():
        value = F.filter(regex='%s_%s_urban' % (P,k)).sum(axis=1).mean(axis=0)
        new_df_urban.set_value(P, k, value)
        value = F.filter(regex='%s_%s_ag' % (P,k)).sum(axis=1).mean(axis=0)
        new_df_ag.set_value(P, k, value)



plot_clustered_stacked([new_df_urban, new_df_ag],['urban','ag'])
plt.show()
