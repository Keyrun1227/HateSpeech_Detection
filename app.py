from flask import Flask, render_template, request
import pickle

model=pickle.load(open('Hate_Pred','rb'))
vector=pickle.load(open('c_vectorization','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/kiran', methods=['POST'])
def home():

    speech= request.form['speech']
    
    predict=model.predict(vector.transform([speech]))[0]
    
    return render_template('home.html',prediction_speech='The Speech is {}'.format(predict))
    
  
if __name__ == "__main__":
    app.run(debug=True)