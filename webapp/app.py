from flask import Flask, render_template, request, send_file
from wtforms import Form, TextAreaField, validators
import pickle, sqlite3, os, numpy as np
from vectorizer import vect
import uuid, json
import facerecognition.recognize as REC
import base64
import io
import logging, logging.config
import sys
 
LOGGING = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO'
    }
}
 
logging.config.dictConfig(LOGGING)

cur_dir=os.path.dirname(__file__)
clf=pickle.load(open(os.path.join(cur_dir, 
                        'pickled_objects', 'classifier.pkl'), 'rb'))

def classify(document):
    label={0:'negative', 1:'positive'}
    X=vect.transform([document])
    y=clf.predict(X)[0]
    p=clf.predict_proba(X).max()
    return label[y], p

def train(document, y):
    X=vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(document, y):
    conn=sqlite3.connect('reviews.sqlite')
    c=conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"\
        " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()


app=Flask(__name__)

class ReviewForm(Form):
    review=TextAreaField('', [validators.DataRequired(), 
                                validators.length(min=15)])

@app.route('/')
def index():
    form=ReviewForm(request.form)
    return render_template('review.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form=ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review=request.form['review']
        y, p=classify(review)
        return render_template('results.html',
            content=review,
            prediction=y,
            probability=round(p*100,2))
    return render_template('review.html', form=form)



@app.route('/feedback', methods=['POST'])
def feedback():
    feedback=request.form['feedback_button']
    review=request.form['review']
    prediction=request.form['prediction']

    inv_label={'negative': 0, 'positive': 1}
    y=inv_label[prediction]
    if feedback=='Incorrect':
        y=int(not(y))
    train(review, y)
    sqlite_entry(review, y)
    return render_template('feedback.html')


@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        logging.info('Hello1 '+ str(request.files))

        file = request.files['file']
        logging.info('Hello2')
        extension = os.path.splitext(file.filename)[1]
        if not extension: 
            extension = ".jpg"
        logging.info('Hello3')
        f_name = str(uuid.uuid4()) + extension
        logging.info('Hello4')
        f_path = os.path.join("./uploaded_images/", f_name)
        logging.info('Hello5')
        file.save(f_path)
        logging.info('Hello6')
        f_outpath = os.path.join("./rec_images/", f_name)
        logging.info("fPath is " + f_path + ", Recfile is " + f_outpath)
        REC.recognize(f_path, f_outpath)
        logging.info('Hello7')
        data = open(f_outpath, "rb").read()
        data = base64.b64encode(data)
        return send_file(io.BytesIO(data), mimetype='image/jpg')

@app.route('/recognize_submit')
def recognize_submit():
    return render_template('recognize_submit.html')

@app.route('/jpeg_camera/<path:path>')
def send_js(path):
    return send_file('jpeg_camera\\'+ path)

if __name__ == "__main__":
    app.run(debug=True)
