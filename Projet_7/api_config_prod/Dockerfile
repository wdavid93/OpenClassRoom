# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM python:3.9-slim

# Récupération fichiers
COPY . /app

# Répertoire de travail
WORKDIR /app

# Dépendance pour XGBoost
RUN apt-get update
RUN apt-get install -y libgomp1

# Using pip:
RUN pip install altair
RUN pip install argh
RUN pip install astor
RUN pip install attrs
RUN pip install backcall
RUN pip install base58
RUN pip install bleach
RUN pip install blinker
RUN pip install boto3
RUN pip install botocore
RUN pip install certifi
RUN pip install chardet
RUN pip install Click
RUN pip install colorama
RUN pip install cycler
RUN pip install decorator
RUN pip install defusedxml
RUN pip install docutils
RUN pip install entrypoints
RUN pip install enum-compat
RUN pip install future
RUN pip install idna
RUN pip install importlib-metadata
RUN pip install ipykernel
RUN pip install ipython
RUN pip install ipython-genutils
RUN pip install itsdangerous
RUN pip install jedi
RUN pip install Jinja2
RUN pip install jmespath
RUN pip install joblib
RUN pip install json5
RUN pip install jsonschema
RUN pip install jupyter-client
RUN pip install jupyter-core
RUN pip install jupyterlab
RUN pip install jupyterlab-server
RUN pip install kiwisolver
RUN pip install MarkupSafe
RUN pip install matplotlib
RUN pip install mistune
RUN pip install more-itertools
RUN pip install nbconvert
RUN pip install nbformat
RUN pip install notebook
RUN pip install numpy
RUN pip install pandas
RUN pip install pandocfilters
RUN pip install parso
RUN pip install pathtools
RUN pip install pickleshare
RUN pip install Pillow
RUN pip install prometheus-client
RUN pip install prompt-toolkit
RUN pip install protobuf
RUN pip install Pygments
RUN pip install pyparsing
RUN pip install pyrsistent
RUN pip install python-dateutil
RUN pip install pytz
RUN pip install pywinpty
RUN pip install PyYAML
RUN pip install pyzmq
RUN pip install requests
RUN pip install s3transfer
RUN pip install scikit-learn
RUN pip install scipy
RUN pip install seaborn
RUN pip install Send2Trash
RUN pip install six
RUN pip install streamlit
RUN pip install terminado
RUN pip install testpath
RUN pip install toml
RUN pip install toolz
RUN pip install tornado
RUN pip install traitlets
RUN pip install tzlocal
RUN pip install urllib3
RUN pip install validators
RUN pip install watchdog
RUN pip install wcwidth
RUN pip install webencodings
RUN pip install Werkzeug
RUN pip install xgboost
RUN pip install zipp

# Déclaration du port d'entrée à l'app depuis l'extérieur du container
EXPOSE 80

# Déplacement des fichiers de configuration de streeamlit
# dans un répertoire .streamlit
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
WORKDIR /app

# Lance streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["ocdsp7_pkl.py"]