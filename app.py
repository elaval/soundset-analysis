from __future__ import print_function
import os
import ast
import json
from urllib.parse import unquote_plus

# Timers for asyc process
from threading import Timer, Event

# Firebase management
import firebase_admin
from firebase_admin import credentials, firestore

import lib.firebaseStorage as firebaseStorage

import lib.s3storage as s3storage
import lib.soundset as soundset

from flask import Flask, request, jsonify, redirect, url_for, make_response
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
CORS(app) 
api = Api(app)


print(os.listdir("/credentials"))

def ProcessPendingDocs(): 
  jobList = firebaseStorage.getPendingJobs()
  lockedJobs = firebaseStorage.getLockedJobs()
  for job in jobList:
      print(job)
  if len(jobList) > 0:
    nextJob = jobList[0]
    firebaseStorage.logLastActivity("START PROCESS {}".format(nextJob))
    success = PorcessJob(nextJob)
    if not success:
      firebaseStorage.logLastActivity("ERROR PROCESSING {}".format(nextJob))
    else:
      firebaseStorage.logLastActivity("SUCCESS PROCESSING {}".format(nextJob))

    ProcessPendingDocs()
  else:
    msg = "NO PENDING JOBS - {} LOCKED JOBS".format(len(lockedJobs))
    firebaseStorage.logLastActivity(msg)
    print(msg)

def PorcessJob(id):
  firebaseStorage.lockJob(id)
  job = firebaseStorage.getJobDict(id)
  fileKey = unquote_plus(job['key']) # The file name might have plus (+) signs instead of spaces WE NEED TO CLEAN THIS ON THE SOURCE 
  success = False
  try:
    results =  soundset.processFile(fileKey)
    firebaseStorage.logJob(id,results)
    firebaseStorage.removeJob(id)
    firebaseStorage.updateDailyUsage(results)
    success = True
  except Exception as e:
    firebaseStorage.logError({
      "key": fileKey,
      "id": id,
      "error": str(e)
    })
    results = str(e)

  print(results)
  return success


# Function called every 10 seconds
def f(f_stop):
    # do something here ...
    ProcessPendingDocs()

    if not f_stop.is_set():
        # call f() again in 60 seconds
        Timer(300, f, [f_stop]).start()

f_stop = Event()
# start calling f now and every 60 sec thereafter
f(f_stop)

# stop the thread when needed
#f_stop.set()

class SoundSetAPI(Resource):
    def get(self):
        ProcessPendingDocs()
        return {"output": "DONE"}


api.add_resource(SoundSetAPI, '/trigger')

 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')