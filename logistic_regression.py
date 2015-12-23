# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author: marlon

"""


import numpy
import pandas
import statsmodels.api as sm
import seaborn
import statsmodels.formula.api as smf 

pandas.set_option('display.float_format', lambda x:'%.2f'%x)
data = pandas.read_csv('/Users/utpl/Documents/RegressionModelingInPractice/nesarc_pds.csv', low_memory=False)


### setting variables to work with to numeric

data['IDNUM'] = pandas.to_numeric(data['IDNUM'], errors='coerce')
data['TAB12MDX'] = pandas.to_numeric(data['TAB12MDX'], errors='coerce')
data['SOCPDLIFE'] = pandas.to_numeric(data['SOCPDLIFE'], errors='coerce')
data['MAJORDEPLIFE'] = pandas.to_numeric(data['MAJORDEPLIFE'], errors='coerce')
data['ANTISOCDX2'] = pandas.to_numeric(data['ANTISOCDX2'], errors='coerce')


# SUBSET DATA

#sub1=data[(data['AGE']<=25) & (data['CHECK321']==1) & (data['S3AQ3B1']==1) & 
#(data['IDNUM']!=20346) & (data['IDNUM']!=36471) & (data['IDNUM']!=28724)]

##CREATE BINARY NICOTINE DEPENDENCE VARIABLE 

def NICOTINEDEP(x):
    if x['TAB12MDX'].any() == 1:
        return 1
    else:
        return 0
sub1['NICOTINEDEP'] = sub1.apply (lambda x: NICOTINEDEP (x), axis=1)
print(pandas.crosstab(sub1['TAB12MDX'], sub1['NICOTINEDEP']))

# LOGISTIC REGRESSION SOCIAL PHOBIA

#lreg1 = smf.logit(formula = 'NICOTINEDEP ~ SOCPDLIFE + MAJORDEPLIFE', data = sub1).fit()
lreg1 = smf.logit(formula = 'TAB12MDX ~ SOCPDLIFE + MAJORDEPLIFE', data = data).fit()
print(lreg1.summary())

## odds ratios

print('Odds Ratios')
print(numpy.exp(lreg1.params))

params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print(numpy.exp(conf))