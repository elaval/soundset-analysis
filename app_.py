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
import boto3

from lib.soundset import storeFile, getSpectrum, batchesToPlainArray, buildSamples, buildPredictions
import ntpath

import firebase_admin
from firebase_admin import credentials, firestore

import datetime
import re

import ast
import json

from threading import Timer, Event

dummy = os.environ.get('GOOGLE_SERVICE_ACCOUNT')
file = open("./credentials/firebase_cred.json", "w")
json.dump(ast.literal_eval(dummy), file)
file.close()

"""
with open('gsa.json', 'w') as outfile:
    json.dump(ast.literal_eval(os.environ.get('GOOGLE_SERVICE_ACCOUNT')),
                  outfile)
"""
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials/firebase_cred.json"

S3_KEY =os.environ.get('S3_KEY')
S3_SECRET=os.environ.get('S3_SECRET')
FIREBASE_KEY = os.environ.get('FIREBASE_PRIVATE_KEY')
FIREBASE_EMAIL = os.environ.get('FIREBASE_EMAIL')

#cred = credentials.Certificate("./credentials/soundset-abffd-firebase-adminsdk-sz6e6-659bee9e48.json")
default_app = firebase_admin.initialize_app()

""" default_app = firebase_admin.initializeApp({
  credential: admin.credential.cert({
    projectId: 'soundset-abffd',
    clientEmail: FIREBASE_EMAIL,
    privateKey: FIREBASE_PRIVATE_KEY
  }),
}) """

db = firestore.client()

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app) 
api = Api(app)


print(tf.__version__)

debug = []

targetSoundset = "demo"

modelFile = "./models/enel/model.h5" 

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model(modelFile)

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

testFileDir = "./inputs/%s/testMP3" % targetSoundset


#Get Labels
labels = []

file = open("./models/enel/labels.txt", "r")
for line in file:
    labels.append(line.strip())

print(labels) 

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


def listS3Folder(bucket, prefix, id, secret):
    s3 = boto3.resource('s3', aws_access_key_id=id,
        aws_secret_access_key=secret)
    s3Bucket = s3.Bucket(name=bucket)

    output = []
    for obj in s3Bucket.objects.filter(Prefix=prefix):
        file = ntpath.basename(obj.key)
        if file:
            output.append(file)


    return output

def downloadS3File(bucket, prefix, file, id, secret):
    s3r = boto3.resource('s3', aws_access_key_id=id,
    aws_secret_access_key=secret)

    fileName = prefix + file
    buck = s3r.Bucket(bucket)
    destFile = os.path.join("./inputs/uploaded/",file)

    return buck.download_file(fileName,destFile)
    

def uplaodS3File(bucket, prefix, file, id, secret):
    s3r = boto3.resource('s3', aws_access_key_id=id,
    aws_secret_access_key=secret)

    fileName = prefix + file
    buck = s3r.Bucket(bucket)
    srcFile = os.path.join("./outputs/",file)

    return buck.upload_file(srcFile,'enel/output/'+file)


def csvResponse(data):
    si = io.StringIO()
    cw = csv.writer(si, delimiter='\t')
    cw.writerows(data)
    response = make_response(si.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=export.tsv"
    response.headers["Content-type"] = "text/tsv"
    return response

def csvToFile(name,data):
    with open('./outputs/'+name+".tsv", mode='w') as output_file:
        tsv_writer = csv.writer(output_file, delimiter='\t')
        tsv_writer.writerows(data)


def checkNewFiles(bucket, prefix):
    soundFiles = listS3Folder(bucket, prefix, S3_KEY, S3_SECRET)
    predictionFiles = listS3Folder(bucket, "output", S3_KEY, S3_SECRET)

    processedNames = {}
    for file in predictionFiles:
        fileBaseName = ntpath.basename(file)
        (name,ext) = os.path.splitext(fileBaseName)
        processedNames[name] = True

    newFiles = []
    for file in soundFiles:
        fileBaseName = ntpath.basename(file)
        (name,ext) = os.path.splitext(fileBaseName)
        if name not in processedNames:
            newFiles.append(file)

    return newFiles

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


def ProcessPendingDocs():
    docs = db.collection(u'newFiles').where(u'bucket', u'==', "soundset").get()
        
    output = []
    for doc in docs:
        output.append(doc.id)
    print(output)

# Function called every 10 seconds
def f(f_stop):
    # do something here ...
    ProcessPendingDocs()

    if not f_stop.is_set():
        # call f() again in 60 seconds
        Timer(60, f, [f_stop]).start()

f_stop = Event()
# start calling f now and every 60 sec thereafter
f(f_stop)

# stop the thread when needed
#f_stop.set()

todos = {}
class Test(Resource):
    def get(self):
        docs = db.collection(u'newFiles').where(u'bucket', u'==', "soundset").get()
        
        output = []
        for doc in docs:
            output.append(doc.id)
        return output

class TodoSimple(Resource):
    def get(self):
        output = process()
        return {"output": str(output)}

    def put(self):
        """
        parse = request.get_json()
        id=S3_KEY
        secret=S3_SECRET
        bucket= parse["bucket"]
        folder = parse["folder"]
        iam = boto3.client('iam')

        try:
            output = listS3Folder(bucket, folder, id, secret)
        except:
            output = "Access denied"
        

        return output
        """
        id=os.environ.get('S3_KEY')
        secret=os.environ.get('S3_SECRET')

        docs = db.collection(u'newFiles').where(u'bucket', u'==', "soundset").get()

        list = []
        docIds = []

        for doc in docs:
            print(u'{} => {}'.format(doc.id, doc.to_dict()))
            docDict = doc.to_dict()
            list.append(docDict['key'])
            docIds.append(doc.id)

        target = list[0]
        docId = docIds[0]


        pattern = re.compile(r"(?P<company>[a-zA-Z0-9 ]+?)/input/(?P<path>.+)")
        m = pattern.search(target)
        company = (m.group('company'))
        path = (m.group('path'))

        fileBaseName = ntpath.basename(path)
        (name,ext) = os.path.splitext(fileBaseName)
        
        destFile = os.path.join("./inputs/uploaded/",fileBaseName)
        destFileWav = os.path.join("./inputs/uploaded/",name+".wav")
        bucket = "soundset"
        prefix = company + "/input/" 
        
        downloadS3File("soundset", prefix, path, id, secret)
        
        spectrum = getSpectrum(destFile)

        os.remove(destFile)
        os.remove(destFileWav)

        samples = buildSamples(spectrum)
        predictions = buildPredictions(model, samples, name, labels)

        si = io.StringIO()
        cw = csv.writer(si, delimiter='\t')

        csvToFile(name,predictions)
        uplaodS3File(bucket, prefix, name+".tsv", id, secret)
        cw.writerows(predictions)

        docs = db.collection(u'newFiles').document(docId).delete()

        response = make_response(si.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=export.tsv"
        response.headers["Content-type"] = "text/tsv"
        return docId

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
        predictions = buildPredictions(model, samples)
        response = csvResponse(predictions)
        return response

class S3Demo(Resource):
    def post(self):
        parse = request.get_json()
        id=os.environ.get('S3_KEY')
        secret=os.environ.get('S3_SECRET')
        bucket= parse["bucket"]
        prefix = parse["prefix"]

        si = io.StringIO()
        cw = csv.writer(si, delimiter='\t')


        fileList = checkNewFiles(bucket, prefix)

        for file in fileList:
            fileBaseName = ntpath.basename(file)
            (name,ext) = os.path.splitext(fileBaseName)
            
            destFile = os.path.join("./inputs/uploaded/",fileBaseName)
            destFileWav = os.path.join("./inputs/uploaded/",name+".wav")
            
            downloadS3File(bucket, prefix, file, id, secret)
            
            spectrum = getSpectrum(destFile)

            os.remove(destFile)
            os.remove(destFileWav)

            samples = buildSamples(spectrum)
            predictions = buildPredictions(model, samples, name, labels)

            ref = db.collection(u"processed_files")
            ref.add({
                u"file":fileBaseName,
                u"output":name+".tsv",
                u"samples":len(samples),
                u"date":datetime.datetime.now()
            })

            today = datetime.date.today()
            todayId = "%02d%02d%02d" % (today.year, today.month, today.day)

            totalRef = db.collection(u"summary").document(todayId)

            totalRef.update({
                u'lastUpdate': datetime.datetime.now()
            })
            
            

            csvToFile(name,predictions)
            uplaodS3File(bucket, prefix, name+".tsv", id, secret)
            cw.writerows(predictions)
            

        
        response = make_response(si.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=export.tsv"
        response.headers["Content-type"] = "text/tsv"
        return response

        

api.add_resource(Test, '/test')
api.add_resource(TodoSimple, '/elm')
api.add_resource(BatchManager, '/batches')
api.add_resource(VectorManager, '/vectors')
api.add_resource(ClassificationManager, '/classes')
api.add_resource(S3Demo, '/s3')

 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')