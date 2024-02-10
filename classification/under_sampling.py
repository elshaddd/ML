from yellowbrick.target import ClassBalance
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.90],
                           flip_y=0, random_state=1)
# Случайным образом выбираются и удаляются 𝑁 мажоритарных объектов 
undersample = RandomUnderSampler(random_state=0)
X, y = undersample.fit_resample(X, y)

# Связи Томека объединяют близко расположенные объекты различных классов
undersample = TomekLinks()
X, y = undersample.fit_resample(X, y)

# Правило сосредоточенного ближайшего соседа
undersample = CondensedNearestNeighbour(n_neighbors=1)
X, y = undersample.fit_resample(X, y)

# Односторонний сэмплинг (One-side sampling, one-sided selection— OSS). Сочетание двух предыдущих подходов.
undersample = OneSidedSelection()
X3, y3 = undersample.fit_resample(X, y)

# Правило «очищающего» соседа (Neighborhood cleaning rule— NCR)
# Все объекты классифицируются по правилу трех ближайших соседей (3-NN). Удаляются мажоритарные объекты: получившие правильную метку класса и соседи миноритарных объектов, неверно классифицированных.
undersample = NeighbourhoodCleaningRule()
X, y = undersample.fit_resample(X, y)
