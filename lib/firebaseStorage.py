import os
import ast
import json
import datetime

# Firebase management
import firebase_admin
from firebase_admin import credentials, firestore

JOBS_PENDING_COLLECTION = u'jobsPending'
JOBS_DONE_COLLECTION = u'jobsDone'
DAILY_USAGE_COLLECTION = u'dailyUsage'
LOGS_COLLECTION = u'logs'
ERRORS_COLLECTION = u'errors'
BUCKET = u'soundset'
UNLOCKED_PROPERTY = u'unlocked'

# Firebase library expects credentials to  be stored in a JSON file
# 
# An environment variable FIREBASE_CREDENTIAL_PATH will specify the 
# location of the credentials file
# If using Docker, we expect the file to be inlcuded on a host location and mounted into the running container
# THE CREDENTIAL SHOULD NOT BE INCLUDED INSIDE THE COMNTAINER
# 

# Initialize app
FIREBASE_CREDENTIAL_PATH = os.environ.get('FIREBASE_CREDENTIAL_PATH')
cred = credentials.Certificate(FIREBASE_CREDENTIAL_PATH)
default_app = firebase_admin.initialize_app(cred)


# Create reference to firestore db
db = firestore.client()

jobsDicts = {}

def getPendingJobs():
  docs = db.collection(JOBS_PENDING_COLLECTION).where(u'bucket', u'==', BUCKET).where(UNLOCKED_PROPERTY, u'==', True).get()
        
  output = []
  for doc in docs:
    jobsDicts[doc.id] = doc.to_dict()
    output.append(doc.id)

  return output

def getLockedJobs():
  docs = db.collection(JOBS_PENDING_COLLECTION).where(u'bucket', u'==', BUCKET).where(UNLOCKED_PROPERTY, u'==', False).get()
        
  output = []
  for doc in docs:
    jobsDicts[doc.id] = doc.to_dict()
    output.append(doc.id)

  return output

# When we start processing a job, we lock it with a mark and a timestamp
# We avoid processing jobs that are recently locked
def lockJob(id):
  lockData = {}
  lockData[UNLOCKED_PROPERTY] = False
  lockData["lockTimeStamp"] = datetime.datetime.now()

  doc = db.collection(JOBS_PENDING_COLLECTION).document(id)
  doc.update(lockData)

def getJobDict(id):
  return jobsDicts[id]

def removeJob(id):
  db.collection(JOBS_PENDING_COLLECTION).document(id).delete()

def logJob(id, summary):
  db.collection(JOBS_DONE_COLLECTION).document(id).set(summary)  
  db.collection(JOBS_DONE_COLLECTION).document(id).update({
    "timestamp": datetime.datetime.now()
  })

def updateDailyUsage(summary):
  today = datetime.date.today()
  todayId = "%02d%02d%02d" % (today.year, today.month, today.day)

  totalRef = db.collection(DAILY_USAGE_COLLECTION).document(todayId)

  updateData = {
    u'lastUpdate': datetime.datetime.now()
  }

  try:
    doc = doc_ref.get()
    docDict = doc.to_dict()
    if "seconds" in docDict:
      updateData["seconds"] = docDict["seconds"] + summary["seconds"]
    else : 
      updateData["seconds"] = summary["seconds"]
    totalRef.update(updateData)
  except:
    updateData["seconds"] = summary["seconds"]
    totalRef.set(updateData)

def logLastActivity(info):
  lastActivityRef = db.collection(LOGS_COLLECTION).document("LAST_ACTIVITY")

  updateData = {
    u'timestamp': datetime.datetime.now(),
    u'info': info
  }

  try:
    lastActivityRef.set(updateData)
  except:
    print("LOG ERROR")

def logError(info):
  errorRef = db.collection(ERRORS_COLLECTION)

  updateData = {
    u'timestamp': datetime.datetime.now(),
    u'info': info
  }
  errorRef.add(updateData)