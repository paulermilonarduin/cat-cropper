@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Installation des dependances...
python -m pip install -e .
if errorlevel 1 (
  echo Echec installation dependances.
  pause
  exit /b 1
)

echo [2/2] Recadrage des photos...
python crop_cat.py
if errorlevel 1 (
  echo Echec du traitement.
  pause
  exit /b 1
)

echo Termine. Resultats dans Pictures\CROPPED
pause
