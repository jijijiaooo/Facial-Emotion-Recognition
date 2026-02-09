#!/bin/bash
# Azure ML Deployment Script - Enhanced Emotion Detection
# This script automates the deployment process

set -e  # Exit on error

echo "======================================================================"
echo "Azure ML Endpoint Deployment - Enhanced Emotion Detection"
echo "======================================================================"

# Check prerequisites
echo -e "\n✅ Checking prerequisites..."

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Install from: https://docs.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi
echo "  ✓ Azure CLI installed"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi
echo "  ✓ Python 3 installed"

# Check if model file exists
MODEL_FILE="models/emotion_enhanced_cnn_20251130_234141.h5"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Model file not found: $MODEL_FILE"
    echo "   Please train the model first: python src/core/train_enhanced_cnn.py"
    exit 1
fi
echo "  ✓ Model file found: $MODEL_FILE"

# Get Azure configuration
echo -e "\n📋 Azure Configuration"
echo "======================================================================"

read -p "Subscription ID: " SUBSCRIPTION_ID
read -p "Resource Group Name: " RESOURCE_GROUP
read -p "Workspace Name: " WORKSPACE_NAME
read -p "Endpoint Name: " ENDPOINT_NAME
read -p "Azure Region [eastus]: " LOCATION
LOCATION=${LOCATION:-eastus}

echo -e "\nConfiguration Summary:"
echo "  Subscription:    $SUBSCRIPTION_ID"
echo "  Resource Group:  $RESOURCE_GROUP"
echo "  Workspace:       $WORKSPACE_NAME"
echo "  Endpoint:        $ENDPOINT_NAME"
echo "  Location:        $LOCATION"

read -p "Proceed with deployment? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

# Login to Azure
echo -e "\n🔐 Logging into Azure..."
az login

# Set subscription
echo -e "\n📌 Setting subscription..."
az account set --subscription "$SUBSCRIPTION_ID"

# Create resource group if it doesn't exist
echo -e "\n📦 Creating resource group (if needed)..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" || true

# Create Azure ML workspace if it doesn't exist
echo -e "\n🏢 Creating Azure ML workspace (if needed)..."
az ml workspace create \
    --name "$WORKSPACE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" || true

# Install Python dependencies for deployment
echo -e "\n📚 Installing Python dependencies..."
pip install -q azure-ai-ml azure-identity

# Run deployment script
echo -e "\n🚀 Deploying model to Azure ML..."
python3 deploy_azure_ml.py \
    --subscription-id "$SUBSCRIPTION_ID" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --endpoint-name "$ENDPOINT_NAME"

# Get endpoint details
echo -e "\n📡 Getting endpoint details..."
ENDPOINT_URL=$(az ml online-endpoint show \
    --name "$ENDPOINT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --query scoring_uri -o tsv)

echo -e "\n======================================================================"
echo "✅ Deployment Complete!"
echo "======================================================================"
echo ""
echo "Endpoint URL: $ENDPOINT_URL"
echo ""
echo "To get your API key, run:"
echo "  az ml online-endpoint get-credentials \\"
echo "    --name $ENDPOINT_NAME \\"
echo "    --resource-group $RESOURCE_GROUP \\"
echo "    --workspace-name $WORKSPACE_NAME"
echo ""
echo "To test your endpoint:"
echo "  python test_azure_ml_endpoint.py \\"
echo "    --endpoint-url \"$ENDPOINT_URL\" \\"
echo "    --api-key \"<your-api-key>\" \\"
echo "    --image \"path/to/test/image.jpg\""
echo ""
echo "======================================================================"
