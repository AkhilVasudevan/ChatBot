from flask import Flask,render_template,jsonify,request
import json
from ChatBot import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat',methods=['POST'])
def chat():
    message=request.form['message']
    resp=ChatBot(message)
    return resp
    
if __name__ == '__main__':
    app.run(debug = True)