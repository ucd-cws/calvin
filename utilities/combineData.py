# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:22:25 2020

@author: Mahesh L Maskey, August 14, 2020
This routine combines the output files from CALVIN for different years merge into one.
"""
import pandas as pd
def read_combine(yearList, param): 
    resultdir=('result/annual/WY%d' % yearList[0])
    F = pd.read_csv(resultdir + '/' + param + '.csv', index_col=0, parse_dates=True)
    
    
    for i in range(yearList[1],yearList[-1]):    
      print('\nNow running WY %d' % i)
      resultdir=('result/annual/WY%d' % i)
      F2 = pd.read_csv(resultdir + '/' + param + '.csv', index_col=0, parse_dates=True)
      F = pd.concat([F, F2], axis=0)
      
    F.to_csv('result/' + param + '.csv', index=True)
    
yearList = [i for i in range(1990, 2005)]
read_combine(yearList, 'dual_lower')
read_combine(yearList, 'dual_node')
read_combine(yearList, 'dual_upper')
read_combine(yearList, 'evaporation')
read_combine(yearList, 'flow')
read_combine(yearList, 'shortage_cost')
read_combine(yearList, 'shortage_volume')
read_combine(yearList, 'storage')
