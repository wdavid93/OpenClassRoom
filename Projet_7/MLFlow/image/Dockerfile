FROM python:3.9

# Using pip:
RUN pip install mlflow

EXPOSE 5001

RUN mkdir /mlflow
WORKDIR /mlflow

#CMD ["mlflow server --port 5001"]
CMD ["python3 -m http.server"]