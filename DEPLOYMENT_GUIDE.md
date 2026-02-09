# 🚀 Azure Web App Deployment Guide

This guide will teach you how to deploy your Flask Emotion Detection API to Azure App Service.

## 📋 What You'll Need

1. **Azure Student Account** (with free credits)
2. **Docker Desktop** (for building container)
3. **Azure CLI** (command-line tool)
4. **About 30 minutes**

---

## 🎯 Method 1: Automated Deployment (Easiest)

### Step 1: Install Prerequisites

#### Install Azure CLI (macOS)
```bash
brew install azure-cli
```

#### Install Docker Desktop
Download from: https://www.docker.com/products/docker-desktop

### Step 2: Make Script Executable
```bash
chmod +x deploy_to_azure.sh
```

### Step 3: Run Deployment Script
```bash
./deploy_to_azure.sh
```

The script will:
- ✅ Check prerequisites
- ✅ Login to Azure
- ✅ Create resource group
- ✅ Create container registry
- ✅ Build and push Docker image
- ✅ Create App Service
- ✅ Deploy your API

### Step 4: Test Your API

After deployment completes, test it:

```bash
# Test health endpoint (replace with your URL)
curl https://your-app-name.azurewebsites.net/health

# Should return:
# {"status":"healthy","service":"emotion-detection-api","version":"1.0.0"}
```

---

## 🔧 Method 2: Manual Deployment (Learning)

If you want to understand each step:

### Step 1: Install Azure CLI

```bash
# macOS
brew install azure-cli

# Windows (PowerShell as Admin)
# Download from: https://aka.ms/installazurecliwindows

# Linux
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### Step 2: Login to Azure

```bash
az login
```

This opens your browser. Login with your student account.

### Step 3: Create Resource Group

```bash
# Create a resource group (logical container for resources)
az group create \
  --name rg-emotion-api \
  --location eastus
```

### Step 4: Create Container Registry

```bash
# Create Azure Container Registry (ACR) to store Docker image
az acr create \
  --resource-group rg-emotion-api \
  --name emotionapiacr \
  --sku Basic

# Enable admin access
az acr update --name emotionapiacr --admin-enabled true
```

### Step 5: Build and Push Docker Image

```bash
# Login to ACR
az acr login --name emotionapiacr

# Build Docker image
docker build -t emotion-api:latest .

# Tag for ACR
docker tag emotion-api:latest emotionapiacr.azurecr.io/emotion-api:v1

# Push to ACR
docker push emotionapiacr.azurecr.io/emotion-api:v1
```

### Step 6: Create App Service Plan

```bash
# Create App Service Plan (defines VM size and pricing)
az appservice plan create \
  --name asp-emotion-api \
  --resource-group rg-emotion-api \
  --is-linux \
  --sku B1
```

**Pricing Tiers:**
- **F1** (Free): 60 min/day CPU limit
- **B1** (Basic): ~$13/month, good for dev/test
- **S1** (Standard): ~$70/month, production-ready

### Step 7: Create Web App

```bash
# Get ACR credentials
ACR_PASSWORD=$(az acr credential show --name emotionapiacr --query "passwords[0].value" -o tsv)

# Create Web App
az webapp create \
  --resource-group rg-emotion-api \
  --plan asp-emotion-api \
  --name emotion-api-yourname \
  --deployment-container-image-name emotionapiacr.azurecr.io/emotion-api:v1

# Configure container
az webapp config container set \
  --name emotion-api-yourname \
  --resource-group rg-emotion-api \
  --docker-custom-image-name emotionapiacr.azurecr.io/emotion-api:v1 \
  --docker-registry-server-url https://emotionapiacr.azurecr.io \
  --docker-registry-server-user emotionapiacr \
  --docker-registry-server-password $ACR_PASSWORD
```

### Step 8: Configure App Settings

```bash
# Set port configuration
az webapp config appsettings set \
  --resource-group rg-emotion-api \
  --name emotion-api-yourname \
  --settings WEBSITES_PORT=8000 PORT=8000
```

### Step 9: Get Your App URL

```bash
az webapp show \
  --name emotion-api-yourname \
  --resource-group rg-emotion-api \
  --query defaultHostName \
  --output tsv
```

Your API is now live at: `https://emotion-api-yourname.azurewebsites.net`

---

## 🧪 Testing Your Deployment

### Test Health Endpoint

```bash
curl https://emotion-api-yourname.azurewebsites.net/health
```

### Test Emotion Detection

```python
import requests
import base64

# Your Azure URL
API_URL = "https://emotion-api-yourname.azurewebsites.net"

# Read test image
with open("test_image.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

# Send request
response = requests.post(
    f"{API_URL}/api/detect",
    json={"image": img_base64}
)

print(response.json())
```

---

## 📱 Connect to Android App

Update your Android app's API configuration:

```kotlin
// ApiClient.kt
object ApiClient {
    // Replace with your Azure URL
    private const val BASE_URL = "https://emotion-api-yourname.azurewebsites.net/"
    
    // ... rest of code
}
```

---

## 📊 Monitor Your App

### View Logs (Real-time)

```bash
az webapp log tail \
  --name emotion-api-yourname \
  --resource-group rg-emotion-api
```

### View in Azure Portal

```bash
az webapp browse \
  --name emotion-api-yourname \
  --resource-group rg-emotion-api
```

### Restart App

```bash
az webapp restart \
  --name emotion-api-yourname \
  --resource-group rg-emotion-api
```

---

## 🔄 Update Your App

When you make changes to your code:

```bash
# 1. Rebuild Docker image
docker build -t emotion-api:latest .

# 2. Tag new version
docker tag emotion-api:latest emotionapiacr.azurecr.io/emotion-api:v2

# 3. Push to ACR
docker push emotionapiacr.azurecr.io/emotion-api:v2

# 4. Update Web App
az webapp config container set \
  --name emotion-api-yourname \
  --resource-group rg-emotion-api \
  --docker-custom-image-name emotionapiacr.azurecr.io/emotion-api:v2

# 5. Restart app
az webapp restart \
  --name emotion-api-yourname \
  --resource-group rg-emotion-api
```

---

## 💰 Cost Management

### View Costs

```bash
# Check your current costs
az consumption usage list \
  --start-date 2026-01-01 \
  --end-date 2026-01-31
```

### Stop App (to save costs)

```bash
# Stop app when not using
az webapp stop \
  --name emotion-api-yourname \
  --resource-group rg-emotion-api

# Start again when needed
az webapp start \
  --name emotion-api-yourname \
  --resource-group rg-emotion-api
```

### Delete Everything

```bash
# Delete entire resource group (removes all resources)
az group delete \
  --name rg-emotion-api \
  --yes
```

---

## 🐛 Troubleshooting

### App won't start

```bash
# View detailed logs
az webapp log tail --name emotion-api-yourname --resource-group rg-emotion-api

# Check container logs
az webapp log download --name emotion-api-yourname --resource-group rg-emotion-api
```

### 503 Service Unavailable

- App is still starting (wait 2-3 minutes)
- Check logs for errors
- Verify Docker image runs locally first

### Model not loading

- Check if `models/` folder was copied to Docker image
- Verify Dockerfile includes: `COPY models/ /app/models/`
- Check file size limits (Azure has 1GB limit per file)

### Out of memory

- Upgrade to higher SKU (B2 or S1)
- Reduce model size or use TFLite

---

## 📚 Additional Resources

- **Azure Portal**: https://portal.azure.com
- **Azure Documentation**: https://docs.microsoft.com/azure
- **Azure Student**: https://azure.microsoft.com/free/students/
- **Pricing Calculator**: https://azure.microsoft.com/pricing/calculator/

---

## ✅ Deployment Checklist

- [ ] Azure CLI installed
- [ ] Docker Desktop installed and running
- [ ] Logged into Azure
- [ ] Resource group created
- [ ] Container registry created
- [ ] Docker image built and pushed
- [ ] App Service created
- [ ] App deployed successfully
- [ ] Health endpoint returns 200 OK
- [ ] Emotion detection endpoint tested
- [ ] Android app updated with new URL
- [ ] Logs monitored for errors

---

## 🎓 What You Learned

1. ✅ How to use Azure CLI
2. ✅ Docker containerization for deployment
3. ✅ Azure Container Registry (ACR)
4. ✅ Azure App Service (Web Apps)
5. ✅ Continuous deployment workflow
6. ✅ Cloud monitoring and logging
7. ✅ Cost management in Azure

---

**Congratulations! Your emotion detection API is now running in the cloud! 🎉**
