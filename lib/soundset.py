import os
from pydub import AudioSegment
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

UPLOAD_DIR = "./inputs/uploaded/"

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