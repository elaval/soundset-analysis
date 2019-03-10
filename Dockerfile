FROM tensorflow/tensorflow:latest-py3
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git 
RUN git clone https://github.com/tensorflow/models.git &&\ 
    cp models/research/audioset/* . &&\ 
    rm -r models
RUN apt-get install -y curl ffmpeg
RUN curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
RUN curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
# RUN cp models/research/audioset/* .
# RUN rm -r models
RUN pip install boto3
RUN pip install --upgrade firebase-admin
COPY . /app
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
