#!/bin/bash
# Deploy Emotion Detection API to Azure Container Instances
# This works with Azure Student subscriptions

set -e

echo "🚀 Deploying Emotion Detection API to Azure Container Instances"
echo "================================================================"

# Configuration
RESOURCE_GROUP="rg-emotion-api"
LOCATION="eastus"
ACR_NAME="emotiondetectacr"
IMAGE_NAME="emotion-api"
IMAGE_TAG="v1"
ACI_NAME="emotion-api-container"

# 1. Create resource group
echo "📦 Step 1: Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# 2. Create Azure Container Registry
echo "📦 Step 2: Creating Azure Container Registry..."
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true

# 3. Login to ACR
echo "🔐 Step 3: Logging into ACR..."
az acr login --name $ACR_NAME

# 4. Build and push image to ACR
echo "🏗️  Step 4: Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .

echo "📤 Step 5: Pushing image to ACR..."
docker tag $IMAGE_NAME:$IMAGE_TAG $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG
docker push $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG

# 5. Get ACR credentials
echo "🔑 Step 6: Getting ACR credentials..."
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv)

# 6. Deploy to Azure Container Instances
echo "🚀 Step 7: Deploying to Azure Container Instances..."
az container create \
  --resource-group $RESOURCE_GROUP \
  --name $ACI_NAME \
  --image $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG \
  --registry-login-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --dns-name-label emotion-api-$(whoami) \
  --ports 8000 \
  --cpu 2 \
  --memory 4 \
  --environment-variables PORT=8000

# 7. Get the URL
echo ""
echo "✅ Deployment complete!"
echo "================================================================"
FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $ACI_NAME --query ipAddress.fqdn --output tsv)
echo "🌐 Your API is available at: http://$FQDN:8000"
echo ""
echo "Test endpoints:"
echo "  Health check: curl http://$FQDN:8000/health"
echo "  API docs: http://$FQDN:8000/"
echo "================================================================"
