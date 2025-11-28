@echo off
REM AI Code Review Platform - Windows Setup Script
echo.
echo ========================================
echo    AI Code Review Platform Setup
echo ========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker not found. Please install Docker Desktop first.
    echo    Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo ✅ Docker found
echo.

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose not found. Please install Docker Compose first.
    pause
    exit /b 1
)

echo ✅ Docker Compose found
echo.

REM Setup AI versions
echo 🤖 Setting up AI versions...
if not exist "ai_versions" mkdir ai_versions
if not exist "ai_versions\v1_stable" mkdir ai_versions\v1_stable\model
if not exist "ai_versions\v1_stable\config" mkdir ai_versions\v1_stable\config
if not exist "ai_versions\v2_experimental" mkdir ai_versions\v2_experimental\model
if not exist "ai_versions\v2_experimental\config" mkdir ai_versions\v2_experimental\config
if not exist "ai_versions\v3_deprecated" mkdir ai_versions\v3_deprecated\model
if not exist "ai_versions\v3_deprecated\config" mkdir ai_versions\v3_deprecated\config

REM Create model info files
echo CodeBERT > ai_versions\v1_stable\model\model_name.txt
echo Llama2 > ai_versions\v2_experimental\model\model_name.txt
echo GPT-3.5-turbo > ai_versions\v3_deprecated\model\model_name.txt

REM Create blocklist
echo Creating model blocklist...
(
echo blocked_models:
echo   - "gpt-3.5-turbo"  # Reason: high_latency
echo   - "mistral-7b"     # Reason: cost
) > ai_versions\blocklist.yaml

echo.
echo ✅ AI versions setup complete!
echo.
echo Available versions:
echo   v1_stable: CodeBERT ^(CPU optimized^)
echo   v2_experimental: Llama2 ^(GPU optimized^)
echo   v3_deprecated: GPT-3.5-turbo ^(Archive^)
echo.
echo Usage:
echo   npm run build:ai    - Build frontend
echo   npm run start:ai    - Start all services
echo   npm run dev:frontend - Frontend only
echo.
echo ========================================
echo.

REM Ask user what to do
set /p choice=What would you like to do?
set /p choice1=Build frontend ^(build^)
set /p choice2=Start all services ^(start^)
set /p choice3=Frontend only ^(dev^)
set /p choice4=Exit ^(exit^)

set /p choice=%choice1%
:loop
if "%choice%"=="" set /p choice=%choice1%
echo.
echo %choice%
echo.
echo ========================================
echo.

if /i "%choice%"=="1" (
    echo 🏗 Building frontend...
    call npm run build:frontend
) else if /i "%choice%"=="2" (
    echo 🚀 Starting all AI services...
    call npm run start:ai
) else if /i "%choice%"=="3" (
    echo 🖥 Starting frontend development server...
    call npm run dev:frontend
) else if /i "%choice%"=="4" (
    echo 👋 Goodbye!
    goto :end
) else (
    echo Invalid choice. Please try again.
    goto :loop
)

:end
pause