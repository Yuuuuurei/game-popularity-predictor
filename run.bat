@echo off
echo ----------------------------------------
echo Mengecek dan membuat virtual environment...
echo ----------------------------------------

:: Cek apakah folder .env ada
if not exist .env (
    echo Membuat virtual environment...
    python -m venv .env
)

:: Aktifkan virtual environment
echo Mengaktifkan virtual environment...
call ".env\Scripts\activate.bat"

:: Pastikan pip tersedia
where pip >nul 2>nul
if errorlevel 1 (
    echo pip tidak ditemukan. Pastikan Python terinstall.
    exit /b
)

:: Install requirements.txt
echo ----------------------------------------
echo Menginstall dependencies dari requirements.txt...
echo ----------------------------------------
pip install -r requirements.txt

:: Jalankan streamlit dengan PYTHONPATH=.
echo ----------------------------------------
echo Menjalankan Streamlit Dashboard...
echo ----------------------------------------

:: Jalankan perintah dalam PowerShell agar bisa set PYTHONPATH
powershell -Command "$env:PYTHONPATH='.'; streamlit run app/dashboard.py"

echo.
echo ----------------------------------------
echo Dashboard selesai dijalankan.
pause
