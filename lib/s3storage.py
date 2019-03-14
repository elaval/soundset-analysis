import boto3
import os

S3_KEY =os.environ.get('S3_KEY')
S3_SECRET=os.environ.get('S3_SECRET')

s3 = boto3.resource('s3', aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET)


def listS3Folder(bucket, prefix):
    s3Bucket = s3.Bucket(name=bucket)

    output = []
    for obj in s3Bucket.objects.filter(Prefix=prefix):
        file = ntpath.basename(obj.key)
        if file:
            output.append(file)
    return output

def downloadS3File(bucket, prefix, file, destPath):
    fileName = prefix + "/" + file
    buck = s3.Bucket(bucket)

    return buck.download_file(fileName,destPath)
    

def uplaodS3File(bucket, prefix, file, srcPath):

    fileName = prefix + file
    buck = s3.Bucket(bucket)

    return buck.upload_file(srcPath,fileName)


def downloadModel(bucket, domain, destPathModel, destPathLabels):
    buck = s3.Bucket(bucket)

    modelKey = domain + "/model/model.h5"
    labelsKey = domain + "/model/labels.txt"

    buck.download_file(modelKey,destPath)    
    buck.download_file(labelsKey,destPath)

    return 