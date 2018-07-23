FROM "ubuntu"
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y python-pip

WORKDIR /usr/src/app  
RUN pip install flask
RUN pip install tensorflow  

add templates templates
add uploads	uploads
COPY tf_files/retrained_graph.pb  tf_files/retrained_graph.pb
COPY tf_files/retrained_labels.txt  tf_files/retrained_labels.txt
COPY classifyPainter.py classifyPainter.py
CMD [ "python", "./classifyPainter.py" ]