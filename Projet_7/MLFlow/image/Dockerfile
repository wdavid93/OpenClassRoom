FROM python:3.9

RUN mkdir /mlflow
WORKDIR /mlflow

EXPOSE 5001

# Using pip:
RUN pip install --no-cache mlflow

# CMD [ "python", "-m", "http.server" ]
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001"]
