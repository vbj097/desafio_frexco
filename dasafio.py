# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:56:42 2023

@author: vbjun
"""

import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as stats
import numpy as np
import statsmodels.tsa.stattools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.utils import tsdisplay, autocorr_plot, decomposed_plot, plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from pmdarima.utils import diff_inv
import seaborn as sns

d = pd.read_excel("Dados.xlsx")

#dados
vendas = [ 870,  868, 1189,  742,  317,  685, 1366, 1213, 1055, 1343,  832,
        240,  235, 1050,  711,  745, 1009,   18,   40,   67,  821,  572,
        429,  638,  106,   54,  144,  814,  679,  712, 1229,  821,  319,
        317, 1317,  807,  923, 1265,  892,  289,  566, 1692, 1097, 1302,
       1405,  945]
dias = pd.date_range('2022-12-06', periods=len(vendas))

#análise descritiva
plt.figure(figsize=(8, 6))
plt.hist(vendas)
plt.show()

sns.set_style("whitegrid")
sns.boxplot(x = vendas)

df = pd.DataFrame(vendas)
df.describe()

serie = pd.Series(data = vendas, index= dias)
plt.figure(figsize=[12, 9.5]);
plt.plot(serie)
plt.title("Vendas de 06/12/2022 à 20/01/2023")
plt.xlabel('Tempo')
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
decomposicao = seasonal_decompose(serie)
decomposicao.plot()

plot_acf(serie)
plot_pacf(serie, lags =10)

#transformação

serie2 = pd.Series(data=np.diff(np.diff(serie)), index=dias[2:46])
decomposicao = seasonal_decompose(serie2)
decomposicao.plot()

#Modelagem

modelo = auto_arima(serie2, trace=True, stepwise = True, seasonal=True)
print("AIC obtido: {}". format(modelo.aic()))
modelo.summary()

#análise de residuos

residuos = modelo.resid()

stats.probplot(residuos, dist='norm', plot=plt)
plt.title('Normal QQ Plot')
plt.show()

#teste de aderência à distribuição normal
stats.shapiro(diff_inv(residuos))

x = range(0,len(residuos),1)
plt.scatter(x,residuos)
plt.axhline(y = 0.0, color = 'r', linestyle = '-')
plt.title("Dispersão dos resíduos")
plt.show()


#previsão


pred = modelo.predict(n_periods=6)

#reajustando a escala
d = pd.date_range('2023-01-20', periods=len(pred)+1)
p = pd.DataFrame(diff_inv(pred), columns=['Previsão'])
p.insert(0,"data", d)
p = p.drop(p.index[0])
p

prev = pd.DataFrame(diff_inv(pred), columns=['Previsão'])
prev.loc[0]=945 
a = pd.concat([serie,prev])
a=a.rename(columns={0:'série'})
a.plot()

#desempenho

v_predito = modelo.predict_in_sample(start=0, end=42)
plt.plot(v_predito)
plt.plot(serie2)
plt.title("Valores preditos e observados")
plt.legend(['predito', 'valor observado'])
plt.show()

