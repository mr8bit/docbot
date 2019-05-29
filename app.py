from flask import Flask
from flask import render_template
from flask import request
from docsim import DocSim
app = Flask(__name__)

from gensim.models.keyedvectors import KeyedVectors
model_path = 'ruwikiruscorpora-nobigrams_upos_skipgram_300_5_2018.vec'
w2v_model = KeyedVectors.load_word2vec_format(model_path)
ds = DocSim(w2v_model=w2v_model, dataset_path="прога курсач.csv")


@app.route('/', methods=['GET', 'POST']) # раз
def hello_world():
    if request.method == 'POST':
        source_doc = ds.text_POS_tag(request.form['question'])
        answer = ds.get_answer(source_doc)
        context = {
            'answer': answer[0],
            'score': answer[1],
            'question': request.form['question']
        }
        return render_template('index.html', context=context)
    else:
        context = {'answer': None}
        return render_template('index.html', context=context)


if __name__ == '__main__':
    app.debug = True
    app.run()
