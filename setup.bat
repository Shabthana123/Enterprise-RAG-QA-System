@echo off
setlocal

echo Setting up RAG System...

:: Step 1: Install uv if not present
echo.
echo Checking for 'uv' installation...
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo ⏳ Installing 'uv'...
    pip install uv
    if %errorlevel% neq 0 (
        echo Failed to install uv. Please ensure Python and pip are installed.
        exit /b 1
    )
) else (
    echo uv is already installed.
)

:: Step 2: Create virtual environment with Python 3.10
echo.
echo Creating virtual environment with Python 3.10...
uv venv .newenv --python 3.10
if %errorlevel% neq 0 (
    echo Failed to create virtual environment. Make sure Python 3.10 is installed.
    exit /b 1
)

:: Step 3: Activate virtual environment and upgrade pip
echo.
echo Activating virtual environment and upgrading pip...
call .newenv\Scripts\activate.bat
python -m ensurepip --upgrade
python -m pip install --upgrade pip

:: Step 4: Install dependencies using uv pip
echo.
echo Installing dependencies from requirements.txt...
uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    exit /b 1
)

:: Step 5: Create .env file if it doesn't exist
echo.
if not exist .env (
    echo GOOGLE_API_KEY=your_gemini_api_key_here > .env
    echo Created .env template. Please update it with your Google API key.
) else (
    echo .env file already exists.
)

echo.
echo Setup complete!
echo.
echo To activate the environment manually later, run:
echo     .newenv\Scripts\activate.bat
echo.
echo Then Follow the Readme files 
echo.

pause