# test_api.py
import sys
import os
import pytest

# Ajoutez le chemin vers le répertoire contenant api.py
# Récupérer le chemin absolu du répertoire parent de votre fichier actuel (là où se trouve le fichier actuel).
current_file_directory = os.path.dirname(__file__)
print("current_file_directory : " , current_file_directory ) 
# Joindre ce chemin avec un chemin relatif pour atteindre le répertoire "api_docker" et obtenir un chemin absolu.
api_docker_directory = os.path.abspath(os.path.join(current_file_directory, "..", "api_docker"))
print("api_docker_directory : " , api_docker_directory ) 
# Insérer ce chemin au début du chemin de recherche Python (sys.path) pour que Python puisse trouver les modules dans ce répertoire.
sys.path.insert(0, api_docker_directory)

# Importez votre module API
import API.py

# Réalisez vos tests ici
@pytest.fixture
def client():
    client = api.app.test_client()
    yield client

def test_home_page(client):
    response = client.get('/')
    assert b"Hello, World!" in response.data

def test_custom_route(client):
    response = client.get('/custom')
    assert b"This is a custom route" in response.data

if __name__ == "__main__":
    pytest.main()