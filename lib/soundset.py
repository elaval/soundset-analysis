import os
from pydub import AudioSegment
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import numpy as np
import re
import lib.s3storage as s3storage
import lib.firebaseStorage as firebaseStorage
import csv
import io
import logging

logger = logging.getLogger('root')

UPLOAD_DIR = "../inputs/uploaded/"
BUCKET = "soundset"

MODEL_FILE = "./models/enel/model.h5" 
MODEL_FOLDER = "./models/enel" 
LABELS_FILE = "./models/enel/labels.txt" 
DOMAIN = "colbun"

remoteModelFolder = os.path.join("model",DOMAIN)

s3storage.downloadS3File(BUCKET, remoteModelFolder, "model.h5", MODEL_FILE)
s3storage.downloadS3File(BUCKET, remoteModelFolder, "labels.txt", LABELS_FILE)

class NeuralNetwork:
    def __init__(self):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        # the folder in which the model and weights are stored
        #self.model_folder = os.path.join(os.path.abspath("src"), "static")
        self.model_folder = MODEL_FOLDER
        self.model = None
        # for some reason in a flask app the graph/session needs to be used in the init else it hangs on other threads
        with self.graph.as_default():
            with self.session.as_default():
                logging.info("neural network initialised")

    def load(self, modelFile=None):
        """
        :param file_name: [model_file_name, weights_file_name]
        :return:
        """
        with self.graph.as_default():
            with self.session.as_default():
                try:
                    if modelFile is not None:
                        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
                            self.model = load_model(modelFile)

                        self.model.compile(optimizer=tf.train.AdamOptimizer(), 
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])

                    logging.info("Neural Network loaded: ")
                    logging.info('\t' + "Neural Network model: " + modelFile)
                    return True
                except Exception as e:
                    logging.exception(e)
                    return False

    def predict(self, x):
        with self.graph.as_default():
            with self.session.as_default():
                y = self.model.predict(x)
        return y

neuralNetwork = NeuralNetwork()
neuralNetwork.load(MODEL_FILE)



#Get Labels
labels = []

file = open(LABELS_FILE, "r")
for line in file:
    labels.append(line.strip())

print(labels) 

def storeFile(args):
  audioFile = args['file']
  audioFile.save("./inputs/uploaded/uploaded.mp3")

  return os.path.join(UPLOAD_DIR,"uploaded.mp3")

def getSpectrum(file):

    (name,ext) = os.path.splitext(file)
    validAudio = False
    
    if ext == ".ogg":
        audio = AudioSegment.from_ogg(file)
        validAudio = True
        
    if ext == ".mp3":
        audio = AudioSegment.from_mp3(file)
        validAudio = True
        
    if validAudio:
        audio.export(name+".wav", format="wav")

    batch = vggish_input.wavfile_to_examples(name+".wav")

    return batch

def batchesToPlainArray(orig_batches):
    batches = []
    for batch in orig_batches:
        timerecords = []
        for record in batch:
            bins =[]
            for bin in record:
                bins.append(bin)
            timerecords.append(bins)
        batches.append(timerecords)
    return batches


def buildSamples(batch):
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, "vggish_model.ckpt")
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

        [samples] = sess.run([embedding_tensor],
                            feed_dict={features_tensor: batch})

    return samples


def buildPredictions(model, samples, name, labels):
    headers = ["name","num","second","class","classNum"]
    for label in labels:
        headers.append(label)

    csvOutput = [headers]

    predictions = model.predict(samples)
    for ndx, member in enumerate(predictions):
        classNum = np.argmax(member)
        row = [name, ndx, ndx*0.96,labels[classNum], classNum  ]
        for i, label in enumerate(labels): 
            row.append(member[i])
        csvOutput.append(row)

    return csvOutput

def csvToFile(name,data):
    with open('./outputs/'+name+".tsv", mode='w') as output_file:
        tsv_writer = csv.writer(output_file, delimiter='\t')
        tsv_writer.writerows(data)

def sumariseClasses(predictions):
    summary = {}
    for i, prediction in enumerate(predictions):
        if i > 0:
            label = prediction[3]
            if label in summary:
                summary[label] = summary[label] +1
            else:
                summary[label] = 1
    return summary



def processFile(fileKey):
  dirName = os.path.dirname(fileKey)
  baseName = os.path.basename(fileKey)
  (name,ext) = os.path.splitext(baseName)

  print(dirName, name, ext)

  # Assuming a dirname with the format
  # data/<company>/input/subdir
  # We use regexp to extract the parts
  # pattern = re.compile(r"(?P<company>[a-zA-Z0-9 ]+?)/input/(?P<subdir>.+)")
  pattern = re.compile(r"data/(?P<company>[a-zA-Z0-9 ]+?)/input/?(?P<subdir>.*)")

  m = pattern.search(dirName)
  company = (m.group('company'))
  subdir = (m.group('subdir'))

  outputPrefix = os.path.join("data/",company, "output", subdir+"/")
  #company+"/output/"+subdir+"/"
  
  destFile = os.path.join("./inputs/uploaded/",baseName)
  destFileWav = os.path.join("./inputs/uploaded/",name+".wav")
  
  s3storage.downloadS3File(BUCKET, dirName, baseName, destFile)
  
  spectrum = getSpectrum(destFile)

  os.remove(destFile)
  os.remove(destFileWav)

  samples = buildSamples(spectrum)
  predictions = buildPredictions(neuralNetwork, samples, name, labels)

  si = io.StringIO()
  cw = csv.writer(si, delimiter='\t')

  csvToFile(name,predictions)

  tmpCSVFile = './outputs/'+name+".tsv"
  s3storage.uplaodS3File(BUCKET, outputPrefix, name+".tsv", tmpCSVFile)
  os.remove(tmpCSVFile)
  cw.writerows(predictions)

  result = {
    "inputKey": fileKey,
    "outputKey": os.path.join(outputPrefix,name+".tsv"),
    "summary": sumariseClasses(predictions),
    "samples": len(samples),
    "seconds": len(samples)*0.96
  }
  return result