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

MODELS_DIR = "./models"
MODEL_FILENAME = "model.h5"
LABELS_FILENAME = "labels.txt"

MODEL_FILE = "./models/enel/model.h5" 
MODEL_FOLDER = "./models/enel" 
LABELS_FILE = "./models/enel/labels.txt" 
DOMAIN = "colbun"

DOMAINS = firebaseStorage.getDomains()


def retrieveModelsAndLabels():
    for domain in DOMAINS:
        localModelDir = os.path.join(MODELS_DIR,domain)
        if not os.path.exists(localModelDir):
            os.makedirs(localModelDir)
        remoteModelFolder = os.path.join("model",domain)
        model_local_file = "./models/%s/model.h5" % domain
        label_local_file = "./models/%s/labels.txt" % domain
        s3storage.downloadS3File(BUCKET, remoteModelFolder, "model.h5", model_local_file)
        s3storage.downloadS3File(BUCKET, remoteModelFolder, "labels.txt", label_local_file)

        print(domain)

retrieveModelsAndLabels()

#remoteModelFolder = os.path.join("model",DOMAIN)

#s3storage.downloadS3File(BUCKET, remoteModelFolder, "model.h5", MODEL_FILE)
#s3storage.downloadS3File(BUCKET, remoteModelFolder, "labels.txt", LABELS_FILE)

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

def loadModels():
    models = {}

    for domain in DOMAINS:
        modelFile = os.path.join(MODELS_DIR,domain,MODEL_FILENAME)
        if os.path.exists(modelFile):
            models[domain] = NeuralNetwork()
            models[domain].load(modelFile)


        """     if os.path.isdir(MODELS_DIR):
        for i, dirname in enumerate(os.listdir(MODELS_DIR)):
            if dirname != ".DS_Store":
                modelFile = os.path.join(MODELS_DIR,dirname,MODEL_FILENAME)
                if os.path.exists(modelFile):
                    models[dirname] = NeuralNetwork()
                    models[dirname].load(modelFile) """
    
    return models

def loadLabels():
    labels = {}

    for domain in DOMAINS:
        labelFile = os.path.join(MODELS_DIR,domain,LABELS_FILENAME)
        if os.path.exists(labelFile):
            domainLabels = []
            file = open(labelFile, "r")
            for line in file:
                domainLabels.append(line.strip())

            labels[domain] = domainLabels

    # if os.path.isdir(MODELS_DIR):
    #     for i, dirname in enumerate(os.listdir(MODELS_DIR)):
    #         if dirname != ".DS_Store":
    #             labelFile = os.path.join(MODELS_DIR,dirname,LABELS_FILENAME)
    #             if os.path.exists(labelFile):
    #                 domainLabels = []
    #                 file = open(labelFile, "r")
    #                 for line in file:
    #                     domainLabels.append(line.strip())

    #                 labels[dirname] = domainLabels
    
    return labels


models = loadModels()
newLabels = loadLabels()
print(models)
print(newLabels)


#neuralNetwork = NeuralNetwork()
#neuralNetwork.load(MODEL_FILE)



#Get Labels
# labels = []


# file = open(LABELS_FILE, "r")
# for line in file:
#     labels.append(line.strip())

# print(labels) 

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

def csvToFile(filePath,data):
    with open(filePath, mode='w') as output_file:
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

  uploadedTmpDir = "./inputs/uploaded/%s" % company 
  if not os.path.exists(uploadedTmpDir):
    os.makedirs(uploadedTmpDir)
  
  destFile = os.path.join(uploadedTmpDir ,baseName)
  destFileWav = os.path.join(uploadedTmpDir,name+".wav")
  
  s3storage.downloadS3File(BUCKET, dirName, baseName, destFile)
  
  spectrum = getSpectrum(destFile)

  os.remove(destFile)
  os.remove(destFileWav)

  samples = buildSamples(spectrum)
  #predictions = buildPredictions(neuralNetwork, samples, name, labels)
  myModel = models[company]
  myLabels = newLabels[company]
  predictions = buildPredictions(myModel, samples, name, myLabels)

  si = io.StringIO()
  cw = csv.writer(si, delimiter='\t')


  tmpCSVFileDir = "./outputs/%s/" % company
  if not os.path.exists(tmpCSVFileDir):
    os.makedirs(tmpCSVFileDir)

  tmpCSVFile = os.path.join(tmpCSVFileDir ,name+".tsv")

  csvToFile(tmpCSVFile,predictions)

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