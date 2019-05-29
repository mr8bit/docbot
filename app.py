from flask import Flask
from flask import render_template
from flask import request
from docsim import DocSim
app = Flask(__name__)

from gensim.models.keyedvectors import KeyedVectors # импорт w2v загрузчика
model_path = 'ruwikiruscorpora-nobigrams_upos_skipgram_300_5_2018.vec' # путь к w2v модели
w2v_model = KeyedVectors.load_word2vec_format(model_path) # загружаем в память
ds = DocSim(w2v_model=w2v_model, dataset_path="прога курсач.csv") # инициализируем класс и передаем в него данные, модель и путь к файлу с бд


@app.route('/', methods=['GET', 'POST']) # разрешаем работу с этим методом через POST и GET
def hello_world():
    if request.method == 'POST': # если пост
        source_doc = ds.text_POS_tag(request.form['question']) # ищем наиболее похожее обращение
        answer = ds.get_answer(source_doc) # получаем ответ
        context = { # передаем все в шаблон
            'answer': answer[0],
            'score': answer[1],
            'question': request.form['question']
        }
        return render_template('index.html', context=context)
    else:
        context = {'answer': None} # если GET то ответа на вопрос нет
        return render_template('index.html', context=context)


if __name__ == '__main__':
    app.run()
