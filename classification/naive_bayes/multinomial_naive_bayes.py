from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
# Загружаем текстовые данные и преобразуем их в числовые
docs = ['This is a sample document.',
        'Another document to test.', 'A third sample for testing.']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
# Открываем модуль классификации и проводим обучение.
clf = MultinomialNB()
clf.fit(X, [0, 1, 0])
# Предсказываем класс нового текста.
new_doc = ['This is another test document.']
new_X = vectorizer.transform(new_doc)
predicted_class = clf.predict(new_X)
