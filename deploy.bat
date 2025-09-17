@echo off
REM Smart Grid System Windows Deployment Script
REM Comprehensive deployment for Windows environments

setlocal enabledelayedexpansion

REM Default values
set ENVIRONMENT=production
set SKIP_TESTS=false
set FORCE_REBUILD=false

REM Parse command line arguments
:parse_args
if "%1"=="" goto end_parse
if "%1"=="--env" (
    set ENVIRONMENT=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--skip-tests" (
    set SKIP_TESTS=true
    shift
    goto parse_args
)
if "%1"=="--force-rebuild" (
    set FORCE_REBUILD=true
    shift
    goto parse_args
)
if "%1"=="-h" goto show_help
if "%1"=="--help" goto show_help
echo Unknown option: %1
exit /b 1

:show_help
echo Usage: %0 [OPTIONS]
echo.
echo Options:
echo   --env ENVIRONMENT    Set deployment environment (development^|staging^|production)
echo   --skip-tests         Skip running tests before deployment
echo   --force-rebuild      Force rebuild of Docker images
echo   -h, --help          Show this help message
exit /b 0

:end_parse

REM Validate environment
if not "%ENVIRONMENT%"=="development" if not "%ENVIRONMENT%"=="staging" if not "%ENVIRONMENT%"=="production" (
    echo ERROR: Invalid environment: %ENVIRONMENT%. Must be one of: development, staging, production
    exit /b 1
)

echo [%date% %time%] Starting Smart Grid System deployment for %ENVIRONMENT% environment

REM Check prerequisites
echo [%date% %time%] Checking prerequisites...

where docker >nul 2>nul
if errorlevel 1 (
    echo ERROR: Docker is not installed or not in PATH
    exit /b 1
)

where docker-compose >nul 2>nul
if errorlevel 1 (
    echo ERROR: Docker Compose is not installed or not in PATH
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo ERROR: Docker daemon is not running
    exit /b 1
)

echo [%date% %time%] Prerequisites check passed

REM Run tests if not skipped
if "%SKIP_TESTS%"=="false" (
    echo [%date% %time%] Running test suite...
    
    python test_suite.py smoke
    if errorlevel 1 (
        echo ERROR: Smoke tests failed. Deployment aborted.
        exit /b 1
    )
    echo [%date% %time%] All tests passed
) else (
    echo [%date% %time%] WARNING: Skipping tests as requested
)

REM Environment-specific configuration
echo [%date% %time%] Configuring environment: %ENVIRONMENT%

set GRID_ENV=%ENVIRONMENT%

if "%ENVIRONMENT%"=="development" (
    set ANOMALY_THRESHOLD=0.7
    set FORECAST_THRESHOLD=0.6
    set PEER_THRESHOLD=1.2
    set COMPOSE_SERVICE=smart-grid-dev
    set PORT=8051
) else if "%ENVIRONMENT%"=="staging" (
    set ANOMALY_THRESHOLD=0.8
    set FORECAST_THRESHOLD=0.7
    set PEER_THRESHOLD=1.4
    set COMPOSE_SERVICE=smart-grid
    set PORT=8050
) else (
    set ANOMALY_THRESHOLD=0.85
    set FORECAST_THRESHOLD=0.8
    set PEER_THRESHOLD=1.5
    set COMPOSE_SERVICE=smart-grid
    set PORT=8050
)

echo [%date% %time%] Environment configuration:
echo    - Anomaly Threshold: %ANOMALY_THRESHOLD%
echo    - Forecast Threshold: %FORECAST_THRESHOLD%
echo    - Peer Threshold: %PEER_THRESHOLD%
echo    - Service: %COMPOSE_SERVICE%
echo    - Port: %PORT%

REM Create required directories
echo [%date% %time%] Creating required directories...
if not exist data mkdir data
if not exist logs mkdir logs
if not exist models mkdir models
if not exist outputs mkdir outputs

REM Build or rebuild Docker images
if "%FORCE_REBUILD%"=="true" (
    echo [%date% %time%] Force rebuilding Docker images...
    docker-compose build --no-cache
) else (
    echo [%date% %time%] Building Docker images...
    docker-compose build
)

REM Stop existing containers
echo [%date% %time%] Stopping existing containers...
docker-compose down

REM Start the service
echo [%date% %time%] Starting Smart Grid System...
docker-compose up -d %COMPOSE_SERVICE%

REM Wait for service to be ready
echo [%date% %time%] Waiting for service to be ready...
timeout /t 10 /nobreak >nul

REM Health check
echo [%date% %time%] Performing health check...
set HEALTH_CHECK_URL=http://localhost:%PORT%
set MAX_RETRIES=30
set RETRY_COUNT=0

:health_check_loop
if %RETRY_COUNT% geq %MAX_RETRIES% (
    echo ERROR: Service failed to become healthy after %MAX_RETRIES% attempts
    exit /b 1
)

curl -f -s "%HEALTH_CHECK_URL%" >nul 2>nul
if not errorlevel 1 (
    echo [%date% %time%] Service is healthy and responsive
    goto health_check_done
)

set /a RETRY_COUNT=%RETRY_COUNT%+1
echo [%date% %time%] Waiting for service... (attempt %RETRY_COUNT%/%MAX_RETRIES%)
timeout /t 5 /nobreak >nul
goto health_check_loop

:health_check_done

REM Show service status
echo [%date% %time%] Service status:
docker-compose ps

REM Show logs
echo [%date% %time%] Recent logs:
docker-compose logs --tail=20 %COMPOSE_SERVICE%

REM Deployment summary
echo [%date% %time%] Deployment completed successfully!
echo.
echo Deployment Summary:
echo    - Environment: %ENVIRONMENT%
echo    - Service: %COMPOSE_SERVICE%
echo    - URL: %HEALTH_CHECK_URL%
echo    - Status: Running
echo.
echo Management commands:
echo    - View logs: docker-compose logs -f %COMPOSE_SERVICE%
echo    - Stop service: docker-compose down
echo    - Restart: docker-compose restart %COMPOSE_SERVICE%
echo    - Access shell: docker-compose exec %COMPOSE_SERVICE% bash
echo.
echo Smart Grid System is now running at %HEALTH_CHECK_URL%

endlocal