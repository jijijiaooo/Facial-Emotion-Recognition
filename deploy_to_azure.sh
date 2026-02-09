#!/bin/bash
# Azure ML Deployment Script for Emotion Detection
# This script will deploy your model to Azure ML Managed Endpoint

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Azure Emotion Detection API Deployment"
echo "=========================================="
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI not found!${NC}"
    echo "Install it with: brew install azure-cli"
    exit 1
fi
echo -e "${GREEN}✅ Azure CLI found${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found!${NC}"
    echo "Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo -e "${GREEN}✅ Docker found${NC}"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running!${NC}"
    echo "Please start Docker Desktop"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

echo ""

# Step 2: Login to Azure
echo -e "${YELLOW}Step 2: Login to Azure...${NC}"
echo "Opening browser for Azure login..."
az login

echo ""

# Step 3: Set variables
echo -e "${YELLOW}Step 3: Configure deployment settings...${NC}"

# Get unique name from user
read -p "Enter a unique name for your app (e.g., emotion-api-yourname): " APP_NAME
if [ -z "$APP_NAME" ]; then
    echo -e "${RED}❌ App name cannot be empty${NC}"
    exit 1
fi

# Default settings
RESOURCE_GROUP="rg-${APP_NAME}"
LOCATION="eastus"
ACR_NAME="${APP_NAME//-/}acr"  # Remove hyphens for ACR name
APP_SERVICE_PLAN="asp-${APP_NAME}"
WEB_APP_NAME="${APP_NAME}"
SKU="B1"  # Basic tier - good for students (~$13/month)

echo ""
echo "Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  Container Registry: $ACR_NAME"
echo "  App Service Plan: $APP_SERVICE_PLAN"
echo "  Web App: $WEB_APP_NAME"
echo "  SKU: $SKU (Basic tier)"
echo ""

read -p "Continue with these settings? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Deployment cancelled"
    exit 0
fi

echo ""

# Step 4: Create Resource Group
echo -e "${YELLOW}Step 4: Creating resource group...${NC}"
if az group show --name "$RESOURCE_GROUP" &> /dev/null; then
    echo -e "${GREEN}✅ Resource group already exists${NC}"
else
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
    echo -e "${GREEN}✅ Resource group created${NC}"
fi

echo ""

# Step 5: Create Azure Container Registry
echo -e "${YELLOW}Step 5: Creating Azure Container Registry...${NC}"
if az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    echo -e "${GREEN}✅ ACR already exists${NC}"
else
    az acr create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$ACR_NAME" \
        --sku Basic
    echo -e "${GREEN}✅ ACR created${NC}"
fi

# Enable admin user
az acr update --name "$ACR_NAME" --admin-enabled true
echo -e "${GREEN}✅ ACR admin enabled${NC}"

echo ""

# Step 6: Build and push Docker image
echo -e "${YELLOW}Step 6: Building Docker image...${NC}"
echo "This may take 5-10 minutes..."

# Login to ACR
az acr login --name "$ACR_NAME"

# Build image
docker build -t emotion-api:latest .
echo -e "${GREEN}✅ Docker image built${NC}"

# Tag image
docker tag emotion-api:latest "${ACR_NAME}.azurecr.io/emotion-api:v1"

# Push to ACR
echo "Pushing image to Azure Container Registry..."
docker push "${ACR_NAME}.azurecr.io/emotion-api:v1"
echo -e "${GREEN}✅ Image pushed to ACR${NC}"

echo ""

# Step 7: Create App Service Plan
echo -e "${YELLOW}Step 7: Creating App Service Plan...${NC}"
if az appservice plan show --name "$APP_SERVICE_PLAN" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    echo -e "${GREEN}✅ App Service Plan already exists${NC}"
else
    az appservice plan create \
        --name "$APP_SERVICE_PLAN" \
        --resource-group "$RESOURCE_GROUP" \
        --is-linux \
        --sku "$SKU"
    echo -e "${GREEN}✅ App Service Plan created${NC}"
fi

echo ""

# Step 8: Create Web App
echo -e "${YELLOW}Step 8: Creating Web App...${NC}"

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

if az webapp show --name "$WEB_APP_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    echo -e "${GREEN}✅ Web App already exists, updating...${NC}"
    az webapp config container set \
        --name "$WEB_APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --docker-custom-image-name "${ACR_NAME}.azurecr.io/emotion-api:v1" \
        --docker-registry-server-url "https://${ACR_NAME}.azurecr.io" \
        --docker-registry-server-user "$ACR_USERNAME" \
        --docker-registry-server-password "$ACR_PASSWORD"
else
    az webapp create \
        --resource-group "$RESOURCE_GROUP" \
        --plan "$APP_SERVICE_PLAN" \
        --name "$WEB_APP_NAME" \
        --deployment-container-image-name "${ACR_NAME}.azurecr.io/emotion-api:v1"
    
    # Configure container settings
    az webapp config container set \
        --name "$WEB_APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --docker-custom-image-name "${ACR_NAME}.azurecr.io/emotion-api:v1" \
        --docker-registry-server-url "https://${ACR_NAME}.azurecr.io" \
        --docker-registry-server-user "$ACR_USERNAME" \
        --docker-registry-server-password "$ACR_PASSWORD"
fi

echo -e "${GREEN}✅ Web App configured${NC}"

echo ""

# Step 9: Configure app settings
echo -e "${YELLOW}Step 9: Configuring app settings...${NC}"
az webapp config appsettings set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$WEB_APP_NAME" \
    --settings WEBSITES_PORT=8000 PORT=8000

echo -e "${GREEN}✅ App settings configured${NC}"

echo ""

# Step 10: Get app URL
echo -e "${YELLOW}Step 10: Getting deployment information...${NC}"
APP_URL=$(az webapp show --name "$WEB_APP_NAME" --resource-group "$RESOURCE_GROUP" --query defaultHostName --output tsv)

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Your API is deployed at:"
echo -e "${GREEN}https://${APP_URL}${NC}"
echo ""
echo "Test endpoints:"
echo "  Health: https://${APP_URL}/health"
echo "  Detect: https://${APP_URL}/api/detect"
echo ""
echo "View logs:"
echo "  az webapp log tail --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "Useful commands:"
echo "  # View app in Azure Portal"
echo "  az webapp browse --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "  # Restart app"
echo "  az webapp restart --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "  # Delete resources (to save costs)"
echo "  az group delete --name $RESOURCE_GROUP --yes"
echo ""
echo "=========================================="

# Save deployment info
cat > deployment_info.txt << EOF
Azure Deployment Information
============================
Deployed on: $(date)

Resource Group: $RESOURCE_GROUP
Location: $LOCATION
Container Registry: $ACR_NAME
App Service Plan: $APP_SERVICE_PLAN
Web App: $WEB_APP_NAME

API URL: https://${APP_URL}

Health Check: https://${APP_URL}/health
Detect Endpoint: https://${APP_URL}/api/detect

View Logs:
az webapp log tail --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP

Delete Resources:
az group delete --name $RESOURCE_GROUP --yes
EOF

echo -e "${GREEN}✅ Deployment info saved to deployment_info.txt${NC}"
echo ""
echo "Next steps:"
echo "1. Test the API: curl https://${APP_URL}/health"
echo "2. Update your Android app's BASE_URL to: https://${APP_URL}/"
echo "3. Monitor logs if there are issues"
