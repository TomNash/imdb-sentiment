#!/usr/bin/env python3

from keras.models import load_model
from keras.preprocessing import sequence
from flask import Flask, request
from flask_restful import Resource, Api
import tensorflow as tf
from keras import backend as K
import json

host = 'http://127.0.0.1'
port = 5000

app = Flask(__name__)
api = Api(app)

class ScoreModel(Resource):
    def __init__(self):
        # set file paths
        self.model_file_path = 'models/imdb_sentiment_RNN.h5'
        # create the model from json and load the weights
        self.model = load_model(self.model_file_path)

    def preprocess(self, data):
        '''

        :param data:
        :return:
        '''
        seq_length = self.model.input_shape[1]
        index_from = 3  # word index offset
        with open('data/imdb_word_index.json') as f:
            word_to_id = json.load(f)
        word_to_id = {k: (v + index_from) for k, v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        # Add a start sequence to the string and map unknown keys to {2: "<UNK>"}
        seq = [1] + [word_to_id.get(x.lower() if x not in ('<PAD>', '<START>', '<UNK>') else x, 2) for x in data.split()]
        # If word is outside the range of words used for model, then cast to unknown
        seq = [[x if x <= 5000 else 2 for x in seq]]
        # Pad sequence
        input_sequence = sequence.pad_sequences(seq, maxlen=seq_length, padding='pre')
        return input_sequence

    def predict(self, data):
        '''

        :param data:
        :return:
        '''
        input_sequence = self.preprocess(data)
        preds = self.model.predict(input_sequence)
        return 'Positive review' if preds[0][0] >= 0.5 else 'Negative review'


@app.route('/')
def api_root():
    """

    :return:
    """
    return 'RestAPI for predictive model, to test'


@app.route('/echo', methods=['POST'])
def echo():
    """

    :return:
    """
    json_data = request.get_json()
    text = json_data.get('text')
    return 'Rest API Pass through: %s\n' % (text)


@app.route('/score', methods=['POST'])
def score_model():
    """

    :return:
    """
    config = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 0})
    session = tf.Session(config=config)
    K.set_session(session)
    json_data = request.get_json()
    text = json_data.get('text')
    s = ScoreModel()
    return s.predict(text) + "\n"


if __name__ == '__main__':
    app.run(host='0.0.0.0')

# test api from command line
# curl -H "Content-Type: application/json" -X POST -d "{\"text\":\"this is a test\"}" http://127.0.0.1:5000/score
