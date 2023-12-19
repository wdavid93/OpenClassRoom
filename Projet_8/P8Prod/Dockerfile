FROM tensorspark

# Ajouts de la liste des bibliothèques python
COPY requirements.txt /home/jovyan/

# RUN python3 -m pip install -r requirements.txt

RUN pip install jupyterhub==1.5.0
RUN pip install typing-extensions
RUN pip install pillow
RUN pip install pyarrow
RUN pip install boto3
RUN pip install s3fs
RUN pip install fsspec
