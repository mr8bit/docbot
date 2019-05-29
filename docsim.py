import numpy as np
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords


class DocSim(object):
    def __init__(self, w2v_model, dataset_path):
        self.w2v_model = w2v_model
        self.stop_words = stopwords.words("russian") # загрузка стоп слов
        self.dataset = pd.read_csv(dataset_path, sep="\t") # чтение базы данных с вопросами и ответами
        self.dataset.columns = ["question", "answer"] # устанавливаем имена колонок
        self.dataset['question'] = self.dataset['question'].apply(lambda x: self.text_POS_tag(x)) # чистим текст и добавляем часть речи

    def word_post_tag(self, word):
        """Преобразовывает слова в их нормальную форму и выводит часть речи"""
        morph = pymorphy2.MorphAnalyzer()
        p = morph.parse(word.lower())[0]
        return "{}_{}".format(p.normal_form, p.tag.POS)

    def text_POS_tag(self, text):
        """Размечает документ в документ слов и часть речи"""
        words = [word for word in text.lower().split() if word not in self.stop_words]
        lists = [self.word_post_tag(word) for word in words]
        return ' '.join(lists)

    def get_docs(self):
        return list(self.dataset['question'])

    def vectorize(self, doc):
        """Преобразование слова в вектор, в данном документе"""
        doc = doc.split()
        word_vecs = []
        for word in doc:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                print(word, "not found")
                pass
        # Вектор документ это среднее значение всех векторов
        vector = np.mean(word_vecs, axis=0)
        return vector

    def _cosine_sim(self, vecA, vecB):
        """Поиск косинусного подобидия между двумя векторами."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_similarity(self, target_docs, threshold=0):
        """Рассчитывает и возвращает оценки сходства между данным исходным документом и всеми
        документы."""
        source_vec = self.vectorize(target_docs)
        results = []
        for doc in self.get_docs():
            target_vec = self.vectorize(doc)
            sim_score = self._cosine_sim(source_vec, target_vec)
            if sim_score > threshold:
                results.append({
                    'score': sim_score,
                    'doc': doc
                })
            # сортируем по score
            results.sort(key=lambda k: k['score'], reverse=True)
        return results

    def get_answer(self, question):
        sim_score = self.calculate_similarity(question)
        most_sim = sim_score[0]['doc']
        # ищем первый наиболее похожий документ в датасете
        # находим и выводим ответ на этот вопрос
        return (list(self.dataset.loc[self.dataset["question"] == most_sim]['answer'])[0], sim_score[0]['score'])
