#!/bin/bash

# Deploy Emotion Detection API to Azure App Service (Web App for Containers)
# This provides better scalability and management than ACI

set -e

echo "========================================================="
echo "Azure App Service Deployment - Emotion Detection API"
echo "========================================================="

# Configuration
az logout
az login --use-device-codeRESOURCE_GROUP="emotion-detection-rg"
LOCATION="eastus"
REGISTRY_NAME="emotiondetectionacr"
IMAGE_NAME="emotion-detection-api"
TAG="latest"
APP_SERVICE_PLAN="emotion-api-plan"
WEB_APP_NAME="emotion-detection-api-${RANDOM}"

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

ACR_LOGIN_SERVER=$(az acr show --name $REGISTRY_NAME --query loginServer -o tsv)
echo "   Login server: $ACR_LOGIN_SERVER"

# Build and push Docker image
echo ""
echo "Step 4: Build and Push Docker Image to ACR"
echo "------------------------------------------"

# Use ACR build task (builds in Azure)
cd "$(dirname "$0")"

az acr build \
    --registry $REGISTRY_NAME \
    --image ${IMAGE_NAME}:${TAG} \
    --file Dockerfile.api \
    .

echo "✅ Image built and pushed to ACR"

# Create App Service Plan (Linux with containers)
echo ""
echo "Step 5: Create App Service Plan"
echo "-------------------------------"
az appservice plan create \
    --name $APP_SERVICE_PLAN \
    --resource-group $RESOURCE_GROUP \
    --is-linux \
    --sku B2 \
    --output table

echo "✅ App Service Plan created: $APP_SERVICE_PLAN"

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query passwords[0].value -o tsv)

# Create Web App
echo ""
echo "Step 6: Create Web App for Containers"
echo "-------------------------------------"
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --name $WEB_APP_NAME \
    --deployment-container-image-name ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${TAG} \
    --output table

echo "✅ Web App created: $WEB_APP_NAME"

# Configure Web App container settings
echo ""
echo "Step 7: Configure Web App"
echo "------------------------"
az webapp config container set \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${TAG} \
    --docker-registry-server-url https://${ACR_LOGIN_SERVER} \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD

# Set environment variables
az webapp config appsettings set \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        PORT=8000 \
        WEBSITES_PORT=8000 \
        PYTHONUNBUFFERED=1

echo "✅ Web App configured"

# Enable continuous deployment (optional)
echo ""
echo "Step 8: Enable Continuous Deployment"
echo "------------------------------------"
az webapp deployment container config \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --enable-cd true

WEBHOOK_URL=$(az webapp deployment container show-cd-url \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query CI_CD_URL -o tsv)

# Configure ACR webhook
az acr webhook create \
    --registry $REGISTRY_NAME \
    --name ${WEB_APP_NAME}webhook \
    --actions push \
    --uri $WEBHOOK_URL

echo "✅ Continuous deployment enabled"

# Restart web app
echo ""
echo "Step 9: Restart Web App"
echo "----------------------"
az webapp restart \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP

echo "✅ Web App restarted"

# Get URL
WEB_APP_URL=$(az webapp show \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query defaultHostName -o tsv)

echo ""
echo "========================================================="
echo "✅ Deployment Complete!"
echo "========================================================="
echo ""
echo "Your API is now available at:"
echo "  https://${WEB_APP_URL}"
echo ""
echo "API Documentation (Swagger UI):"
echo "  https://${WEB_APP_URL}/docs"
echo ""
echo "Health Check:"
echo "  https://${WEB_APP_URL}/health"
echo ""
echo "Test the API:"
echo "  curl https://${WEB_APP_URL}/"
echo ""
echo "Resource Details:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Container Registry: $REGISTRY_NAME"
echo "  App Service Plan: $APP_SERVICE_PLAN (B2 tier)"
echo "  Web App: $WEB_APP_NAME"
echo "  Location: $LOCATION"
echo ""
echo "To view logs:"
echo "  az webapp log tail --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "To update the app (rebuild and redeploy):"
echo "  az acr build --registry $REGISTRY_NAME --image ${IMAGE_NAME}:${TAG} --file Dockerfile.api ."
echo "  (Automatic deployment will trigger via webhook)"
echo ""
echo "To delete resources:"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo ""
echo "========================================================="
