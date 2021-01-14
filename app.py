import numpy as np
from flask import Flask,render_template
import tensorflow as tf
import pickle
import re
def cleaning_documents(articles):

      news = articles.replace('\n',' ')
      news = re.sub('[^\u0980-\u09FF]',' ',str(news)) #removing unnecessary punctuation

      stp = open('bangla_stop_words.txt','r', encoding= 'unicode_escape').read().split()
      result = news.split()
      news = [word.strip() for word in result if word not in stp ]
      news =" ".join(news)
      return news
app = Flask(__name__)

new_model = tf.keras.models.load_model('shagoto1.h5')
class_names = ['abroad',  'capital',
       'court', 'covid19-update', 'cricketworldcup2019',  'economy',
       'education',  'entertainment',
       'national', 'politics',  'scienceandtechnology', 'sports',
        'wholecountry', 'worldnews']
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    final_features =int_features[0]

    cleaned_news = cleaning_documents(final_features)
    seq= loaded_tokenizer.texts_to_sequences([cleaned_news])
    padded = pad_sequences(seq, value=0.0,padding='post', maxlen= 300 )
    pred = new_model.predict(padded)
    m=class_names[np.argmax(pred)]
    result11={}
    cat={'0':'abroad', '1': 'capital','2':
       'court', '3':'covid19-update', '4':'cricketworldcup2019', '5':'economy',
       '6':'education', '7':'entertainment',
       '8':'national', '9':'politics',  '10':'scienceandtechnology', '11':'sports',
       '12':'wholecountry', '13':'worldnews'}
    count=0
    for i in pred[0]:
        result11[cat[str(count)]]=float("{:.2f}".format((i*100)))
        count+=1
    ssssd=sorted(result11.items(), key=lambda x: x[1], reverse=True)
    count1=0
    msg=''
    for i in ssssd:
        if float(i[1]) > 0 and count1<5:
            #print(i[0]+' Percentage :'+str(i[1]))
            msg=msg+i[0]+' :'+str(i[1])+'%'+'\n'
            count1+=1
    return render_template('index.html', prediction_text='Category: {} '.format(m),prediction_text2='{}'.format(msg))



if __name__ == "__main__":
    app.run(debug=True)
