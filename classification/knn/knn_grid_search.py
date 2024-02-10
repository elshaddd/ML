from sklearn.model_selection import GridSearchCV
grid_params = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
               'weights': ['uniform', 'distance'],
               'metric': ['minkowski', 'euclidean', 'manhattan']}
knn = KNeighborsClassifier()
gs = GridSearchCV(knn, grid_params, scoring='accuracy', refit=True)
# Перебираем параметры и визуализируем лучший результат:
g_res = gs.fit(X_tr_std, y_tr)
print(g_res.best_params_)
{'metric': 'minkowski', 'n_neighbors': 4, 'weights': 'distance'}
