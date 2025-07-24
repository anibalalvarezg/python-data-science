import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.linear_model import LassoCV
import pingouin as pg
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.stats.api as sms
import sklearn.linear_model as skl_lm
from sklearn.linear_model import HuberRegressor

df = pd.read_csv('C:/Users/Hamdi/Desktop/UScereal.csv')
print(df.head())

#Muestras train y test
X_train, X_test, y_train, y_test = train_test_split(
    df, df['calories'], test_size=0.2, random_state=42
)

sns.pairplot(X_train, height = 1.5)
plt.show()#grafico mosaico

# 2. Identificación del modelo

#En Python no he podido encontrar algoritmos iterativos de tipo forward, stepwise, backward o Fisher con AIC, BIC y tests de Fisher satisfactorios (sólo R^2 ajustado...).
#Vamos a proceder con correlaciones y correlaciones parciales y LASSO

# Correlación de Pearson
X_train.corr()
# Correlación de Spearman
X_train.corr(method="spearman")
# Correlación de Kendall
X_train.corr(method="kendall")

#correlaciones parciales
pg.partial_corr(data=X_train, x='protein', y='calories', covar=['fat','sodium','fibre','carbo','sugars','potassium'], method='spearman')
#protein eliminado del modelo

pg.partial_corr(data=X_train, x='fat', y='calories', covar=['protein','sodium','fibre','carbo','sugars','potassium'], method='spearman')
#fat se queda en el modelo

pg.partial_corr(data=X_train, x='sodium', y='calories', covar=['protein','fat','fibre','carbo','sugars','potassium'], method='spearman')
#sodium eliminado del modelo

pg.partial_corr(data=X_train, x='fibre', y='calories', covar=['protein','fat','sodium','carbo','sugars','potassium'], method='spearman')
#fibre eliminado del modelo

pg.partial_corr(data=X_train, x='carbo', y='calories', covar=['protein','fat','sodium','fibre','sugars','potassium'], method='spearman')
#carbo se queda en el modelo

pg.partial_corr(data=X_train, x='sugars', y='calories', covar=['protein','fat','sodium','fibre','carbo','potassium'], method='spearman')
#sugars se queda en el modelo

pg.partial_corr(data=X_train, x='potassium', y='calories', covar=['protein','fat','sodium','fibre','carbo','sugars'], method='spearman')
#potassium eliminado del modelo

#IMPORTANTE: Con PEARSON los RESULTADOS son DIFERENTES!! 
#casi todas las correlaciones parciales son significativas...
#esto viene probablemente de los outliers 

#Conclusion: Modelo identificado con las correlaciones, correlaciones parciales.
#Predictores: fat, carbo, sugars

selec_pred = X_train.loc[:, ['fat','carbo','sugars']]

#Identificacion con LASSO
X_train_pred = X_train.drop(columns=['calories'])

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_pred, y_train)

# Get coefficients
coef = pd.Series(lasso.coef_, index=X_train_pred.columns)
coef

# Predictores seleccionados con Lasso
predictores = coef[abs(coef) > 0.1]
print("Predictores seleccionados con Lasso:")
print(predictores)

#Predictores eliminados con Lasso
n_predictores = coef[abs(coef) < 0.1]
print("Predictores eliminados con Lasso:")
print(n_predictores)

#Conclusion: Modelo identificado con LASSO.
#Predictores: protein, fat, carbo, sugars



# 4. Estimación de los modelos de regresión lineal con los predictores seleccionados
#validacion de la colinealidad, outliers y otros puntos

# 4.1 Estimacion OLS del modelo seleccionado con LASSO
X_L = X_train_pred[list(predictores.index.tolist())]
X_const_L = sm.add_constant(X_L)
model_L = sm.OLS(y_train, X_const_L).fit()
print(model_L.summary())
#El test Omnibus es de la normalidad de los residuos.

#4.1.1 colinealidad
X_L.corr()

vif_data = pd.DataFrame()
vif_data["feature"] = X_L.columns
vif_data["VIF"] = [variance_inflation_factor(X_L.values, i) for i in range(X_L.shape[1])]
print(vif_data)

#4.1.2 outliers

#residuos estudentizados y influencia
infl = model_L.get_influence()
student = infl.summary_frame()["student_resid"]
print(student.loc[np.abs(student) > 2])

#leverages/efectos palenca
h_bar = 2 * (model_L.df_model + 1) / model_L.nobs
hat_diag = infl.summary_frame()["hat_diag"]
hat_diag.loc[hat_diag > h_bar]

#influencia
print(infl.summary_frame().loc[1])
print(infl.summary_frame().loc[2])
print(infl.summary_frame().loc[6])
print(infl.summary_frame().loc[31])

#conclusion: tenemos datos outliers e influyentes

#4.1.3 Análisis de heteroscedasticidad

sns.pairplot(X_train, height = 1.5)
plt.show()#grafico mosaico
#Comentario: no hay varianza creciente, son más puntos que se alejan

plt.scatter(model_L.fittedvalues, model_L.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values Plot')
plt.show()
#No hay forma de "trompeta", no hay falta de linealidad
#Hay puntos que se alejan---> mas un problema de outliers

#Test BP:
lm, lm_pvalue, fvalue, f_pvalue = sms.het_breuschpagan(model_L.resid, model_L.model.exog)
print(f'Breusch-Pagan test p-value: {lm_pvalue}')
#La hipótesis de homoscedasticidad es rechazada
#Sin embargo, parece que los datos se alejan mas 
#por outlier que por heteroscedasticidad


# 4.2 Estimacion OLS del modelo seleccionado con correlación parcial
X_const_r = sm.add_constant(selec_pred)
model_r = sm.OLS(y_train, X_const_r).fit()
print(model_r.summary())

# 4.2.1 colinealidad
selec_pred.corr()

vif_data = pd.DataFrame()
vif_data["feature"] = selec_pred.columns
vif_data["VIF"] = [variance_inflation_factor(selec_pred.values, i) for i in range(selec_pred.shape[1])]
print(vif_data)

# 4.2.2 outliers

#residuos estudentizados y influencia
infl = model_r.get_influence()
student = infl.summary_frame()["student_resid"]
print(student.loc[np.abs(student) > 2])

#leverages/efectos palenca
h_bar = 2 * (model_r.df_model + 1) / model_r.nobs
hat_diag = infl.summary_frame()["hat_diag"]
hat_diag.loc[hat_diag > h_bar]

#influencia
print(infl.summary_frame().loc[1])
print(infl.summary_frame().loc[2])
print(infl.summary_frame().loc[6])
print(infl.summary_frame().loc[17])
print(infl.summary_frame().loc[31])
print(infl.summary_frame().loc[50])
print(infl.summary_frame().loc[54])

#Conclusion: hay outliers influyentes también


#4.2.3 Análisis de heteroscedasticidad

plt.scatter(model_r.fittedvalues, model_r.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values Plot')
plt.show()
#No hay forma de "trompeta", no hay falta de linealidad
#Hay puntos que se alejan---> mas un problema de outliers

#Test BP:
lm, lm_pvalue, fvalue, f_pvalue = sms.het_breuschpagan(model_r.resid, model_r.model.exog)
print(f'Breusch-Pagan test p-value: {lm_pvalue}')
#La hipótesis de homoscedasticidad NO es rechazada


# 4.3 Estimacion robusta del modelo seleccionado con LASSO (los dos consideran Huber)
model_L_R = sm.RLM(y_train, X_const_L).fit()
print(model_L_R.summary())

huber = HuberRegressor(epsilon=1.35) # Default epsilon
model_L_R2 =huber.fit(X_const_L, y_train)
print(f"HuberRegressor coefficients: {model_L_R2.coef_}")
print(f"HuberRegressor intercept: {model_L_R2.intercept_}")


# 4.4 Estimacion robusta del modelo seleccionado con correlación parcial (los dos consideran Huber)
model_r_R = sm.RLM(y_train, X_const_r).fit()
print(model_r_R.summary())


model_r_R2 =huber.fit(X_const_r, y_train)
print(f"HuberRegressor coefficients: {model_r_R2.coef_}")
print(f"HuberRegressor intercept: {model_r_R2.intercept_}")



#5.1 validación ocupando el método K-fold (K=5)

scores_r = cross_val_score(model_r_R2, X_const_r, y_train, scoring='neg_mean_squared_error', cv=5)
print(f'MSE medio en validación cruzada: {-np.mean(scores_r):.2f}')

scores_L = cross_val_score(model_L_R2, X_const_L, y_train, scoring='neg_mean_squared_error', cv=5)
print(f'MSE medio en validación cruzada: {-np.mean(scores_L):.2f}')

#5.2 validación ocupando la muestra test

#Modelo seleccionado con LASSO
X_test_pred = X_test.drop(columns=['calories'])
X_L_test = X_test_pred[list(predictores.index.tolist())]
X_const_L_test = sm.add_constant(X_L_test)

pred = model_L_R.predict(X_const_L_test)
MSE = mean_squared_error(y_test, pred)   
print(MSE)


#Modelo seleccionado con correlación parcial
selec_pred_test= X_test.loc[:, ['fat','carbo','sugars']]
X_const_r_test = sm.add_constant(selec_pred_test)

pred = model_r_R.predict(X_const_r_test)
MSE = mean_squared_error(y_test, pred)   
print(MSE)






# 6. Interpretación y predicción de un nuevo individuo y esperanza condicional


#Datos de entrenamiento
print(X_L.head())

#modelo ajustado
model_L_R = sm.RLM(y_train, X_const_L).fit()
print(model_L_R.summary())

# Nuevo individuos (ejemplos ficticios)
v_d= np.ones((5, 1))
x1n = np.linspace(2.5, 5, 5)
Xnew = np.column_stack((v_d,x1n, abs(np.sin(x1n)), (x1n - 0.2) ** 2))

#Prediccion de los nuevos individuos
ynewpred = model_L_R2.predict(Xnew)  # predict out of sample
print(ynewpred)

