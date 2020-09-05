from flask import Flask,render_template,jsonify,request
import uuid
import json
from chatbotProcessor import chatbot

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat',methods=['POST'])
def chat():
    message=request.form['message']
    return str(chatbot.get_response(message))
    
if __name__ == '__main__':
   app.run(debug = True)