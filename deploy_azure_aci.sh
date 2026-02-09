#!/bin/bash

# Deploy Emotion Detection API to Azure Container Instances
# This script builds and deploys the FastAPI application to Azure

set -e

echo "=================================================="
echo "Azure Container Deployment - Emotion Detection API"
echo "=================================================="

# Configuration
RESOURCE_GROUP="emotion-detection-rg"
LOCATION="eastus"
REGISTRY_NAME="emotiondetectionacr"
IMAGE_NAME="emotion-detection-api"
TAG="latest"
CONTAINER_NAME="emotion-api-aci"
DNS_NAME="emotion-detection-api"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first:"
    echo "   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

echo "✅ Azure CLI found"

# Login to Azure
echo ""
echo "Step 1: Azure Login"
echo "-------------------"
az account show &> /dev/null || az login

# Get subscription
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "✅ Using subscription: $SUBSCRIPTION_ID"

# Create resource group
echo ""
echo "Step 2: Create Resource Group"
echo "-----------------------------"
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION \
    --output table

echo "✅ Resource group created: $RESOURCE_GROUP"

# Create Azure Container Registry
echo ""
echo "Step 3: Create Azure Container Registry"
echo "---------------------------------------"
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $REGISTRY_NAME \
    --sku Basic \
    --admin-enabled true \
    --output table

echo "✅ Container registry created: $REGISTRY_NAME"

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $REGISTRY_NAME --query loginServer -o tsv)
echo "   Login server: $ACR_LOGIN_SERVER"

# Build and push Docker image
echo ""
echo "Step 4: Build and Push Docker Image"
echo "-----------------------------------"
echo "Building image locally..."

# Make sure we're in the project root
cd "$(dirname "$0")"

docker build -f Dockerfile.api -t ${IMAGE_NAME}:${TAG} .

echo "✅ Image built successfully"

# Tag image for ACR
docker tag ${IMAGE_NAME}:${TAG} ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${TAG}

# Login to ACR
echo "Logging into Azure Container Registry..."
az acr login --name $REGISTRY_NAME

# Push image to ACR
echo "Pushing image to ACR..."
docker push ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${TAG}

echo "✅ Image pushed to ACR: ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${TAG}"

# Get ACR credentials
echo ""
echo "Step 5: Get ACR Credentials"
echo "---------------------------"
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query passwords[0].value -o tsv)

echo "✅ Retrieved ACR credentials"

# Deploy to Azure Container Instances
echo ""
echo "Step 6: Deploy to Azure Container Instances"
echo "-------------------------------------------"
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${TAG} \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --dns-name-label $DNS_NAME \
    --ports 8000 \
    --cpu 2 \
    --memory 4 \
    --environment-variables PORT=8000 \
    --output table

echo "✅ Container deployed to Azure Container Instances"

# Get FQDN
FQDN=$(az container show \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --query ipAddress.fqdn -o tsv)

echo ""
echo "=================================================="
echo "✅ Deployment Complete!"
echo "=================================================="
echo ""
echo "Your API is now available at:"
echo "  http://${FQDN}:8000"
echo ""
echo "API Documentation (Swagger UI):"
echo "  http://${FQDN}:8000/docs"
echo ""
echo "Health Check:"
echo "  http://${FQDN}:8000/health"
echo ""
echo "Test the API:"
echo "  curl http://${FQDN}:8000/"
echo ""
echo "Resource Details:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Container Registry: $REGISTRY_NAME"
echo "  Container Instance: $CONTAINER_NAME"
echo "  Location: $LOCATION"
echo ""
echo "To view logs:"
echo "  az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""
echo "To delete resources:"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo ""
echo "=================================================="
