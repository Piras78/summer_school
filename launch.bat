@echo off
cd /d "%~dp0"
echo.
echo  Starting Cryo-EM Viewer...
echo.

REM Usa il Python del virtual environment (contiene tutte le dipendenze)
if exist "env\Scripts\python.exe" (
    env\Scripts\python.exe app.py
) else (
    echo ERRORE: virtual environment non trovato.
    echo Esegui prima:  python -m venv env
    echo Poi:           env\Scripts\pip.exe install flask pillow numpy mrcfile
    pause
    exit /b 1
)
pause
