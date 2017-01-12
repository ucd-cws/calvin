
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.cm as cm
import matplotlib
matplotlib.style.use('ggplot')

#Mutliple Stacked Plots - Code from Stackoverflow:

def plot_clustered_stacked(dfall, labels=None, title="Water Supply Portfolio",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    return axe


F = pd.read_csv('flow.csv', index_col=0, parse_dates=True)
# S = pd.read_csv('storage.csv', header=[0,1],index_col=0, parse_dates=True)
# scost = pd.read_csv('shortage_cost.csv', header=[0,1],index_col=0, parse_dates=True)
# svost = pd.read_csv('shortage_volume.csv',header=[0,1],index_col=0, parse_dates=True)
portfolio = pd.read_csv('portfolio.csv', index_col = 0)

new_df_urban = pd.DataFrame(index=portfolio.region.unique())
new_df_ag = pd.DataFrame(index=portfolio.region.unique())

for P in portfolio.region.unique():
    for k in portfolio.supplytype.unique():
        value = F.filter(regex='%s_%s_urban' % (P,k)).sum(axis=1).mean(axis=0)
        new_df_urban.set_value(P, k, value)
        value = F.filter(regex='%s_%s_ag' % (P,k)).sum(axis=1).mean(axis=0)
        new_df_ag.set_value(P, k, value)

# print(new_df)

# new_df.plot.bar(stacked=True)
# plt.show()


plot_clustered_stacked([new_df_urban, new_df_ag],['urban','ag'])
plt.show()

# df.plot.bar()
# plt.xlabel('Region')
# plt.ylabel('FLow [TAF]')
# plt.title('Bar Plot by Region')
# plt.show()
# plt.close('all')

#Water Supply Portfolio Postprocessor



# if you want to plot a particular link or sum of links:
# exports = F['PMP_Banks-D800']+F['PMP_Tracy-D701']
# exports.plot()
# plt.show()

#if you want to search for all links going in/out of a certain node:
#F.filter(regex='SR_SS').plot()


# if you want to get all links associated with surface reservoirs (for example):
#F.filter(regex='SR*').plot()
#plt.show()