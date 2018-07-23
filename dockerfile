FROM "ubuntu"
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y git python-pip

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install flask
RUN pip install tensorflow
RUN apt-get install protobuf-compiler python-pil python-lxml -y




add templates templates
COPY tf_files/retrained_graph.pb  tf_files/retrained_graph.pb
COPY tf_files/retrained_labels.txt  tf_files/retrained_labels.txt
COPY classifyPainter.py classifyPainter.py
CMD [ "python", "./classifyPainter.py" ]