import os
from core.predictor import ZeroDecPredictor
from flask import Flask

#Instantiate the ZeroDecPredictor Class
#This is a one time activity, loads the weights 
#and keeps the model loaded in the memory
zerodec = ZeroDecPredictor()
app = Flask('predictor')

#Create the predictor endpoint
@app.route("/predict", methods=['POST'])
def predict(instances):
    return zerodec.predict(instances)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
