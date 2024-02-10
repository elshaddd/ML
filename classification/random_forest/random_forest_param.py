# Генерируем множество, разделяем его на тестовую и тренировочную части, стандартизируем.
X, y = make_circles(n_samples=500, noise=0.06, random_state=42)
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.3,
                                        random_state=42, stratify=y)
sc = StandardScaler()
sc.fit(X_tr)
X_tr_std = sc.transform(X_tr)
X_t_std = sc.transform(X_t)
# Для числа деревьев от 100 до 200 проверяем результаты работы модели.
error_rate = []
for i in range(100, 200):
    cl = RandomForestClassifier(n_estimators=i)
    cl.fit(X_tr_std, y_tr)
    pred_i = cl.predict(X_t_std)
    error_rate.append(np.mean(pred_i != y_t))

plt.plot(error_rate)
plt.show()
