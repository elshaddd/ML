import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Загружаем dataset, выбираем информативные признаки, разбиваем на тренировочное и тестовое множества, стандартизируем.
data = pd.DataFrame(housing.data, columns=housing.feature_names)
X = data[['MedInc', 'AveBedrms', 'AveRooms', 'Latitude', 'Longitude']]
Xtrain, Xtest, ytain, ytest = train_test_split(
    X, housing.target, test_size=0.2, random_state=42)
sc = StandardScaler()
sc.fit(X)
Xtrain_std = sc.transform(Xtrain)
Xtest_std = sc.transform(Xtest)

# Запускаем модель Ridge, вычисляем метрики RMSE, MAE и 𝑅2.
modelRidge = Ridge(alpha=10.0, fit_intercept=True)
modelRidge.fit(Xtrain_std, ytrain)
yR = modelRidge.predict(Xtest_std)
RMSE_Ridge = mean_squared_error(ytest, yR, squared=False)
MAE_Ridge = mean_absolute_error(ytest, yR)
R2_Riddge = r2_score(ytest, yR)

# Запускаем модель Ridge с итеративной подгонкой, вычисляем метрики.
alphasR = np.linspace(0.1, 10.0, num=1000)
modelRidgeCV = RidgeCV(alphas=alphasR)
modelRidgeCV.fit(Xtrain_std , ytrain)
yRCV = modelRidgeCV.predict(Xtest_std)
RMSE_RidgeCV = mean_squared_error(ytest , yRCV, squared=False)
MAE_RidgeCV= mean_absolute_error(ytest , yRCV)
R2_RiddgeCV = r2_score(ytest , yRCV)

# Аналогично запускаем стандартную модель LASSO и ее вариант с итеративной подгонкой, также вычисляем метрики.
modelLasso = Lasso(alpha=0.3, fit_intercept=True)
modelLasso.fit(Xtrain_std , ytrain)
yL = modelLasso.predict(Xtest_std)
RMSE_Lasso = mean_squared_error(ytest , yL, squared=False)
MAE_Lasso= mean_absolute_error(ytest , yL)
R2_Lasso = r2_score(ytest , yL)
###
alphasL = np.linspace(0.1, 2.0, num=1000)
modelLassoCV = LassoCV(cv=5, random_state=0)
modelLassoCV.fit(Xtrain_std , ytrain)
yLCV = modelLassoCV.predict(Xtest_std)
RMSE_LassoCV = mean_squared_error(ytest , yLCV, squared=False)
MAE_LassoCV= mean_absolute_error(ytest , yLCV)
R2_LassoCV = r2_score(ytest , yLCV)

# Стандартная модель эластичных сетей и ее вариант с итеративной подгонкой.
modelElast = ElasticNet(alpha=0.5, l1_ratio=0.5)
modelElast.fit(Xtrain_std , ytrain)
yE = modelElast.predict(Xtest_std)
RMSE_Elast = mean_squared_error(ytest , yE, squared=False)
MAE_Elast= mean_absolute_error(ytest , yE)
R2_Elast = r2_score(ytest , yE)
###
modelElastCV = ElasticNetCV(cv=5, random_state=0)
modelElastCV.fit(Xtrain_std , y_tr)
yECV = modelElastCV.predict(Xtest_std)
RMSE_ElastCV = mean_squared_error(ytest , yECV, squared=False)
MAE_ElastCV = mean_absolute_error(ytest , yECV)
R2_ElastCV = r2_score(ytest , yECV)

# С помощью библиотеки Yellowbrick визуализируем выбор параметра регуляризации для моделей LASSO, Ridge и эластичной сети с итеративной подгонкой.
from yellowbrick.regressor import AlphaSelection
# RidgeCV
modelR = RidgeCV(alphas=np.linspace(0.1, 10.0, num=1000))
vizR = AlphaSelection(modelR); vizR.fit(Xtrain , ytrrain); vizR.show()
# LassoCV
modelL = LassoCV(alphas=np.linspace(0.1, 2.0, num=1000))
vizL = AlphaSelection(modelL); vizL.fit(Xtrain , ytrain); vizL.show()
# ElasticNetCV
modelE = ElasticNetCV(alphas=np.linspace(0.1, 2.0, num=1000))
vizE = AlphaSelection(modelE); vizE.fit(Xtrain , ytrain); vizE.show()