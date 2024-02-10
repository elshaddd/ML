from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [100, 200],
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': [4, 5, 6, 7, 8],
              'criterion': ['gini', 'entropy']}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='accuracy', refit=True)
# Перебираем параметры и визуализируем лучший результат:
g_res = CV_rfc.fit(X_tr_std, y_tr)
print(g_res.best_params_)
{'criterion': 'gini', 'max_depth': 8, 'max_features': 'auto', 'n_estimators': 114}
