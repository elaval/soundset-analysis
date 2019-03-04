from __future__ import print_function

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

import random

# Helper libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import six
import ffmpeg
import os
from pydub import AudioSegment
import soundfile as soundfile

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
from flask import Flask, request, jsonify, redirect, url_for, make_response
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
from werkzeug.utils import secure_filename
import werkzeug
import io
import csv

from lib.soundset import storeFile, getSpectrum, batchesToPlainArray, buildSamples


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app) 
api = Api(app)


print(tf.__version__)

debug = []

targetSoundset = "demo"

modelFile = "./models/celulosa_v0.2.0.h5" 

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model(modelFile)

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

testFileDir = "./inputs/%s/testMP3" % targetSoundset


#Get Labels
labels = []
jointdir = "./inputs/%s/labels" % targetSoundset 
dirname = jointdir
if os.path.isdir(jointdir):
    for i, dirname in enumerate(os.listdir(jointdir)):  
      labels.append(dirname)

def getSamples(file): 
    batch = vggish_input.wavfile_to_examples(file)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, "vggish_model.ckpt")
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

        [testSamples] = sess.run([embedding_tensor],
                            feed_dict={features_tensor: batch})

    return testSamples

def analyse(allSamples):
  output = [["file", "num", "class", "seconds", "percent"]]
  for [name, samples] in allSamples:
    test = np.array(samples)
    predictions = model.predict(test)
    for ndx, member in enumerate(predictions):
      output.append([name, ndx,  np.argmax(member), ndx*0.96, member[np.argmax(member)]])
  return output

def reformat(data):
    output = []
    for ndx, member in enumerate(data):
      output.append("%s\t%s\t%s\t%s\t" % (ndx,  np.argmax(member), ndx*0.96, member[np.argmax(member)]))

def process():
    allSamples  = []
    if os.path.isdir(testFileDir):

      for i, filename in enumerate(os.listdir(testFileDir)):  
        (name,ext) = os.path.splitext(filename)
        validAudio = False
        
        if ext == ".ogg":
          audio = AudioSegment.from_ogg(os.path.join(testFileDir,filename))
          validAudio = True
          
        if ext == ".mp3":
          debug.append(filename)
          audio = AudioSegment.from_mp3(os.path.join(testFileDir,filename))
          validAudio = True
          
        if validAudio:

          audio.export(os.path.join(testFileDir,name+".wav"), format="wav")
          data, samplerate = soundfile.read(os.path.join(testFileDir,name+".wav"))
          soundfile.write(os.path.join(testFileDir,name+".wav"), data, samplerate, subtype='PCM_16')

          testFileWav = os.path.join(testFileDir,name+".wav")
          samples = getSamples(testFileWav)            
          allSamples.append([name, samples])
      

    debug.append(str(allSamples))
    output = analyse(allSamples)

    return output





def csvResponse(data):
    si = io.StringIO()
    cw = csv.writer(si, delimiter='\t')
    cw.writerows(data)
    response = make_response(si.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=export.tsv"
    response.headers["Content-type"] = "text/tsv"
    return response





def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_whale():
    return "Whale, Hello there!"


# PATCH
# DON'T KNOW the reason but we need to call mode.predict at least once 
# Outside the API methods!!
# beforeusing it in the API method
s = (1,128)
dummySample = np.zeros(s)
model.predict(dummySample)


todos = {}

class TodoSimple(Resource):
    def get(self):
        output = process()
        return {"output": str(output)}

    def put(self):
        data = [[1,2], [3,4]]


        return csvResponse(data)

    def post(self):

        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        audioFile = args['file']
        audioFile.save("./inputs/demo/testMP3/uploaded.mp3")
        
        output = process()
        return csvResponse(output)

class BatchManager(Resource):
    def get(self):
        output = process()
        return {"output": str(output)}

    def put(self):
        data = [[1,2], [3,4]]


        return csvResponse(data)

    def post(self):

        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        filename = storeFile(args)
        output = getSpectrum(filename)

        return output.tolist()

class VectorManager(Resource):
    # NOTE useful to convert from json array to np.arrays
    #b_new = json.loads(obj_text)
    #a_new = np.array(b_new)
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        filename = storeFile(args)
        output = getSpectrum(filename)

        batches = batchesToPlainArray(output)
        samples = buildSamples(batches)
        response = jsonify(samples.tolist())
        response.status_code = 200 # or 400 or whatever
        return response

class ClassificationManager(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        filename = storeFile(args)
        spectrum = getSpectrum(filename)
        samples = buildSamples(spectrum)

        csvOutput = [["num", "class", "seconds", "percent"]]
        predictions = model.predict(samples)
        for ndx, member in enumerate(predictions):
            csvOutput.append([ndx,  np.argmax(member), ndx*0.96, member[np.argmax(member)]])

        response = csvResponse(csvOutput)
        return response

api.add_resource(TodoSimple, '/elm')
api.add_resource(BatchManager, '/batches')
api.add_resource(VectorManager, '/vectors')
api.add_resource(ClassificationManager, '/classes')

 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')