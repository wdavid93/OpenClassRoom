# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM python:3.7.6-slim-buster

# Récupération fichiers
COPY . /app

# Répertoire de travail
WORKDIR /app

# Dépendance pour XGBoost
RUN apt-get update
RUN apt-get install -y libgomp1

# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Using pip:

RUN python3 -m pip install -r requirements.txt

# Déclaration du port d'entrée à l'app depuis l'extérieur du container
EXPOSE 80

# Déplacement des fichiers de configuration de streeamlit
# dans un répertoire .streamlit

RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
WORKDIR /app

# Lance streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["api_dashboard.py"]