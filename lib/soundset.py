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

UPLOAD_DIR = "./inputs/uploaded/"

modelFile = "./models/celulosa_v0.2.0.h5" 

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model(modelFile)

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


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


def buildPredictions(model, samples):
    csvOutput = [["num", "class", "seconds", "percent"]]
    predictions = model.predict(samples)
    for ndx, member in enumerate(predictions):
        csvOutput.append([ndx,  np.argmax(member), ndx*0.96, member[np.argmax(member)]])

    return csvOutput