
if os.path.isdir(dirname):
    for i, dirname in enumerate(os.listdir(dirname)):
      if os.path.isdir(os.path.join(sourcedir,dirname)):
        #Create output dirs
        newDestDir = os.path.join(destdir,dirname)
        if not os.path.exists(newDestDir):
          os.makedirs(newDestDir)
          
        jointDestDir = os.path.join(jointdir,dirname)
        if not os.path.exists(jointDestDir):
          os.makedirs(jointDestDir)
          
        subdir = os.path.join(sourcedir,dirname)
        jointAudio = None
        
        for i, filename in enumerate(os.listdir(subdir)):
          (name,ext) = os.path.splitext(filename)
          if ext == ".mp3":
            print(name)
            audio = AudioSegment.from_mp3(os.path.join(subdir,filename))
            
            if jointAudio is None:
              jointAudio = audio
            else:
              jointAudio = jointAudio + audio
            
            wavFileName = os.path.join(newDestDir, name+".wav")
            audio.export(wavFileName, format="wav")
          
        jointFileName = os.path.join(jointDestDir, "audio.wav")
        jointAudio.export(jointFileName, format="wav")
        data, samplerate = soundfile.read(jointFileName)
        soundfile.write(jointFileName, data, samplerate, subtype='PCM_16')

allBatches = []
labels = []
dirname = jointdir
if os.path.isdir(jointdir):

    for i, dirname in enumerate(os.listdir(jointdir)):  
      labels.append(dirname)
      subdir = os.path.join(jointdir,dirname)
      for i, filename in enumerate(os.listdir(subdir)):
        wavFile = os.path.join(subdir,filename)
        batch = vggish_input.wavfile_to_examples(wavFile)
        print(dirname,"%d batches" % len(batch))
        
        allBatches.append(batch)
        
# Load training & test data

# Prepare a postprocessor to munge the model embeddings.
pproc = vggish_postprocess.Postprocessor("vggish_pca_params.npz")


with tf.Graph().as_default(), tf.Session() as sess:
  # Define the model in inference mode, load the checkpoint, and
  # locate input and output tensors.
  vggish_slim.define_vggish_slim(training=False)
  vggish_slim.load_vggish_slim_checkpoint(sess, "vggish_model.ckpt")
  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)
  
  trainSamplesPerClass = []

  # Run inference and postprocessing.
  for batch in allBatches:
    [train_batch] = sess.run([embedding_tensor],
                               feed_dict={features_tensor: batch})
    trainSamplesPerClass.append(train_batch)
    

  
  #postprocessed_batch = pproc.postprocess(embedding_batch)
  
 
for batch in trainSamplesPerClass:
  test = np.array(batch)
  predictions = model.predict(test)

  for ndx, member in enumerate(predictions):
    print("%s\t%s\t%s\t%s\t" % (ndx,  np.argmax(member), ndx*0.96, member[np.argmax(member)]))
