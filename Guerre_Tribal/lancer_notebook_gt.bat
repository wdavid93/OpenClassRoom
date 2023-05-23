@echo off

REM Chemin vers l'interpréteur Python (peut varier en fonction de votre installation)
set PYTHON_EXECUTABLE=C:\Users\Zbook\AppData\Local\Microsoft\WindowsApps\python3.exe

REM Chemin vers le répertoire contenant les notebooks
set NOTEBOOK_DIR=C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)\Jupyter Notebook (Anaconda3)

REM Nom du fichier notebook à exécuter
set NOTEBOOK_FILE=C:\Users\Zbook\OpenClassRoom\Projet\Guerre_Tribal\Generalisation_fonction_attaque_village.ipynb

REM Lancement du serveur Jupyter
start "Jupyter Notebook" %PYTHON_EXECUTABLE% -m jupyter notebook --notebook-dir="%NOTEBOOK_DIR%"

REM Attente de quelques secondes pour que le serveur Jupyter démarre
timeout /t 5

REM Exécution du fichier notebook spécifié
start "Execute Notebook" %PYTHON_EXECUTABLE% -m jupyter nbconvert --execute "%NOTEBOOK_DIR%\%NOTEBOOK_FILE%"

REM Pause pour garder la fenêtre du terminal ouverte (optionnel)
pause

