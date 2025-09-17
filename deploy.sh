#!/bin/bash

# Smart Grid System Deployment Script
# Comprehensive deployment for production environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Parse command line arguments
ENVIRONMENT="production"
SKIP_TESTS="false"
FORCE_REBUILD="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --force-rebuild)
            FORCE_REBUILD="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env ENVIRONMENT    Set deployment environment (development|staging|production)"
            echo "  --skip-tests         Skip running tests before deployment"
            echo "  --force-rebuild      Force rebuild of Docker images"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    error "Invalid environment: $ENVIRONMENT. Must be one of: development, staging, production"
fi

log "🚀 Starting Smart Grid System deployment for $ENVIRONMENT environment"

# Check prerequisites
log "🔍 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    error "Docker is not installed or not in PATH"
fi

if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose is not installed or not in PATH"
fi

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    warn "Python is not available for local testing"
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    error "Docker daemon is not running"
fi

log "✅ Prerequisites check passed"

# Run tests if not skipped
if [[ "$SKIP_TESTS" == "false" ]]; then
    log "🧪 Running test suite..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        warn "Python not available, skipping local tests"
        SKIP_TESTS="true"
    fi
    
    if [[ "$SKIP_TESTS" == "false" ]]; then
        if ! $PYTHON_CMD test_suite.py smoke; then
            error "Smoke tests failed. Deployment aborted."
        fi
        log "✅ All tests passed"
    fi
else
    warn "Skipping tests as requested"
fi

# Environment-specific configuration
log "⚙️ Configuring environment: $ENVIRONMENT"

export GRID_ENV="$ENVIRONMENT"

case $ENVIRONMENT in
    development)
        export ANOMALY_THRESHOLD=0.7
        export FORECAST_THRESHOLD=0.6
        export PEER_THRESHOLD=1.2
        COMPOSE_SERVICE="smart-grid-dev"
        PORT=8051
        ;;
    staging)
        export ANOMALY_THRESHOLD=0.8
        export FORECAST_THRESHOLD=0.7
        export PEER_THRESHOLD=1.4
        COMPOSE_SERVICE="smart-grid"
        PORT=8050
        ;;
    production)
        export ANOMALY_THRESHOLD=0.85
        export FORECAST_THRESHOLD=0.8
        export PEER_THRESHOLD=1.5
        COMPOSE_SERVICE="smart-grid"
        PORT=8050
        ;;
esac

log "📋 Environment configuration:"
log "   - Anomaly Threshold: $ANOMALY_THRESHOLD"
log "   - Forecast Threshold: $FORECAST_THRESHOLD"
log "   - Peer Threshold: $PEER_THRESHOLD"
log "   - Service: $COMPOSE_SERVICE"
log "   - Port: $PORT"

# Create required directories
log "📁 Creating required directories..."
mkdir -p data logs models outputs

# Build or rebuild Docker images
if [[ "$FORCE_REBUILD" == "true" ]]; then
    log "🔨 Force rebuilding Docker images..."
    docker-compose build --no-cache
else
    log "🔨 Building Docker images..."
    docker-compose build
fi

# Stop existing containers
log "🛑 Stopping existing containers..."
docker-compose down

# Start the service
log "🌟 Starting Smart Grid System..."
docker-compose up -d $COMPOSE_SERVICE

# Wait for service to be ready
log "⏳ Waiting for service to be ready..."
sleep 10

# Health check
log "🔍 Performing health check..."
HEALTH_CHECK_URL="http://localhost:$PORT"
MAX_RETRIES=30
RETRY_COUNT=0

while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    if curl -f -s "$HEALTH_CHECK_URL" > /dev/null 2>&1; then
        log "✅ Service is healthy and responsive"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [[ $RETRY_COUNT -eq $MAX_RETRIES ]]; then
        error "Service failed to become healthy after $MAX_RETRIES attempts"
    fi
    
    log "⏳ Waiting for service... (attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep 5
done

# Show service status
log "📊 Service status:"
docker-compose ps

# Show logs
log "📋 Recent logs:"
docker-compose logs --tail=20 $COMPOSE_SERVICE

# Deployment summary
log "🎉 Deployment completed successfully!"
log ""
log "📋 Deployment Summary:"
log "   - Environment: $ENVIRONMENT"
log "   - Service: $COMPOSE_SERVICE"
log "   - URL: $HEALTH_CHECK_URL"
log "   - Status: Running"
log ""
log "🔧 Management commands:"
log "   - View logs: docker-compose logs -f $COMPOSE_SERVICE"
log "   - Stop service: docker-compose down"
log "   - Restart: docker-compose restart $COMPOSE_SERVICE"
log "   - Access shell: docker-compose exec $COMPOSE_SERVICE bash"
log ""
log "✨ Smart Grid System is now running at $HEALTH_CHECK_URL"