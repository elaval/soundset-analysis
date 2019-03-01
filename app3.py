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
import os
from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/app/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app = Flask(__name__)
CORS(app) 


print(tf.__version__)

targetSoundset = "demo"

def loadmodel():

  modelFile = "/app/models/celulosa_v0.2.0.h5" 

  with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
          model = load_model(modelFile)

testFileDir = "/app/inputs/%s/testMP3" % targetSoundset
allSamples  = []

labels = []
jointdir = "/app/inputs/%s/labels" % targetSoundset 
dirname = jointdir
if os.path.isdir(jointdir):

    for i, dirname in enumerate(os.listdir(jointdir)):  
      labels.append(dirname)


def process():
  if os.path.isdir(testFileDir):

      for i, filename in enumerate(os.listdir(testFileDir)):  
        (name,ext) = os.path.splitext(filename)
        validAudio = False
        
        if ext == ".ogg":
          audio = AudioSegment.from_ogg(os.path.join(testFileDir,filename))
          validAudio = True
          
        if ext == ".mp3":
          audio = AudioSegment.from_mp3(os.path.join(testFileDir,filename))
          validAudio = True
          
        if validAudio:
          print(filename)

          audio.export(os.path.join(testFileDir,name+".wav"), format="wav")
          data, samplerate = soundfile.read(os.path.join(testFileDir,name+".wav"))
          soundfile.write(os.path.join(testFileDir,name+".wav"), data, samplerate, subtype='PCM_16')

          testFileWav = os.path.join(testFileDir,name+".wav")

          batch = vggish_input.wavfile_to_examples(testFileWav)

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
            
            allSamples.append([name, testSamples])
      
  for [name, samples] in allSamples:
    print(name)
    test = np.array(samples)
    predictions = model.predict(test)

    outDir = "/app/inputs/%s/output" % targetSoundset

    outfile = name+".tsv"
    outpath = os.path.join(outDir,outfile)

    with open(outpath, "w") as record_file:
        record_file.write("num\tclass\tsecs\tprob\n")
        for ndx, member in enumerate(predictions):
          printable = False
          if ndx == 0 or ndx == len(predictions)-1 or labels[np.argmax(member)] != "Normal":
            printable = True
            
          if printable:
            print("%s\t%s\t%s\t%s\t" % (ndx,  labels[np.argmax(member)], ndx*0.96, member[np.argmax(member)]))
          record_file.write("%s\t%s\t%s\t%s\n" % (ndx,  labels[np.argmax(member)], ndx*0.96, member[np.argmax(member)]))
            
        print()

 



def analyse():
  output = []
  for [name, samples] in allSamples:
    print(name)
    test = np.array(samples)
    predictions = model.predict(test)
    for ndx, member in enumerate(predictions):
      output.append("%s\t%s\t%s\t%s\t" % (ndx,  np.argmax(member), ndx*0.96, member[np.argmax(member)]))
  return output

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_whale():
    return "Whale, Hello there!"

@app.route('/test')
def analize_():
    output = analyse()
    return str(output)

@app.route('/analize', methods=['GET','POST'])
def login():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
        return str(request.files)
    else:
        return "NO POST"

 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')