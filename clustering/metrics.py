from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import rand_score

# Индекс Рэнда
rand_score(l0, l1)

# Индекс Жаккара
jaccard_score(l0, l1, average=None)

# Индекс Фоулкса–Мэллоуза
fowlkes_mallows_score(l0, l1)

# Индекс Дэвиcа–Болдуина
davies_bouldin = davies_bouldin_score(X, l1)
