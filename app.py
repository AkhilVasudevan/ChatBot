from flask import Flask,render_template,jsonify,request
import uuid
import json
from chatbotProcessor import chatbot
import uuid

app = Flask(__name__)

@app.route('/')
def index():
    customer_id=str(uuid.uuid4())
    data={"Chat":[]}
    with open(customer_id+".json","w+") as f:
        json.dump(data,f)
    f.close()
    return render_template('index.html',customer_id=customer_id)

@app.route('/api/chat',methods=['POST'])
def chat():
    message=request.form['message']
    customer_id=request.form['customer_id']
    resp=str(chatbot.get_response(message))
    with open(customer_id+".json") as f:
        data = json.load(f)
        temp={
                "user":message,
                "vipin":resp
            }
        data["Chat"].append(temp)
    f.close()
    with open(customer_id+".json","w+") as f:
        json.dump(data,f)
    f.close()
    return resp
    
if __name__ == '__main__':
   app.run(debug = True)