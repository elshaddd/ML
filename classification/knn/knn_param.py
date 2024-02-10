import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# Генерируем множество, разделяем его на тестовую и тренировочную части, стандартизируем.
X, y = make_circles(n_samples=500, noise=0.06, random_state=42)
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.3,
                                        random_state=42, stratify=y)
sc = StandardScaler()
sc.fit(X_tr)
X_tr_std = sc.transform(X_tr)
X_t_std = sc.transform(X_t)
# Для 𝑘 ∈ 1 ∶ 40 проверяем результаты работы модели: график ошибок классификации в зависимости от 𝑘 на рисунке 11:b. Можно взять 𝑘 ∈ 5 ∶ 13.
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_tr_std, y_tr)
    pred_i = knn.predict(X_t_std)
    error_rate.append(np.mean(pred_i != y_t))

plt.plot(error_rate)
plt.show()
