# Azure Deployment Guide - Emotion Detection API

Complete guide for deploying the Facial Emotion Recognition system to Microsoft Azure using FastAPI and Docker.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Local Development](#local-development)
5. [Azure Deployment Options](#azure-deployment-options)
6. [Step-by-Step Deployment](#step-by-step-deployment)
7. [Testing the API](#testing-the-api)
8. [Monitoring and Logs](#monitoring-and-logs)
9. [Cost Optimization](#cost-optimization)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This deployment setup containerizes the Enhanced Hybrid Emotion Detection model and deploys it as a REST API on Azure. The system uses:

- **FastAPI**: High-performance web framework for the REST API
- **Docker**: Containerization for consistent deployment
- **Azure Container Registry (ACR)**: Private Docker image registry
- **Azure Container Instances (ACI)** or **Azure App Service**: Hosting options

## ✅ Prerequisites

### Required Tools

1. **Docker Desktop**
   ```bash
   # Install from: https://www.docker.com/products/docker-desktop
   # Verify installation
   docker --version
   docker-compose --version
   ```

2. **Azure CLI**
   ```bash
   # macOS
   brew install azure-cli
   
   # Linux
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   
   # Windows
   # Download from: https://aka.ms/installazurecliwindows
   
   # Verify installation
   az --version
   ```

3. **Azure Subscription**
   - Active Azure subscription
   - Sufficient credits/budget
   - Owner or Contributor role

### Required Files

Ensure you have these trained models in your `models/` directory:
- `emotion_enhanced_cnn_*.h5` - Enhanced CNN model
- `shape_predictor_68_face_landmarks.dat` - dlib facial landmarks (auto-downloaded in Docker)

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                     User/Client                          │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/HTTPS
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Azure Load Balancer                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         FastAPI Application (Docker Container)           │
│  ┌────────────────────────────────────────────────┐     │
│  │  • REST API Endpoints                          │     │
│  │  • Image Upload & Processing                   │     │
│  │  • Face Detection (Haar Cascade)              │     │
│  │  • Facial Landmarks (dlib)                    │     │
│  │  • Enhanced Hybrid CNN Model                  │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and status |
| `/health` | GET | Health check for monitoring |
| `/predict` | POST | Predict emotion from single image |
| `/predict/batch` | POST | Predict emotions from multiple images |
| `/emotions` | GET | Get list of supported emotions |
| `/model/info` | GET | Get model architecture details |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

---

## 💻 Local Development

### 1. Test Locally with Docker

```bash
# Build the Docker image
docker build -f Dockerfile.api -t emotion-detection-api .

# Run the container
docker run -p 8000:8000 emotion-detection-api

# Or use docker-compose
docker-compose -f docker-compose.api.yml up
```

### 2. Test Locally without Docker

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install API dependencies
pip install -r api/requirements.txt

# Run the API
cd api
python main.py
```

### 3. Test the API

```bash
# Using the test script
python api/test_api.py http://localhost:8000

# Or manually with curl
curl http://localhost:8000/

# Test prediction with an image
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

### 4. Access Swagger UI

Open your browser and navigate to:
- **Local**: http://localhost:8000/docs
- **Azure**: https://your-app-url/docs

---

## ☁️ Azure Deployment Options

### Option 1: Azure Container Instances (ACI)

**Best for**: Simple deployments, development/testing

**Pros**:
- Simple and fast deployment
- Pay per second
- No infrastructure management
- Good for prototypes

**Cons**:
- No auto-scaling
- Limited to single container
- Less suitable for production traffic

**Cost**: ~$30-50/month (2 vCPU, 4GB RAM)

### Option 2: Azure App Service (Web App for Containers)

**Best for**: Production deployments, scalable applications

**Pros**:
- Built-in load balancing
- Auto-scaling capabilities
- Continuous deployment
- Better monitoring and logging
- Custom domains and SSL

**Cons**:
- Slightly more complex setup
- Higher minimum cost

**Cost**: ~$55-75/month (B2 tier: 2 cores, 3.5GB RAM)

---

## 🚀 Step-by-Step Deployment

### Option A: Deploy to Azure Container Instances

```bash
# Make the script executable
chmod +x deploy_azure_aci.sh

# Run the deployment
./deploy_azure_aci.sh
```

This script will:
1. Create an Azure Resource Group
2. Create an Azure Container Registry (ACR)
3. Build and push the Docker image to ACR
4. Deploy to Azure Container Instances
5. Provide you with the public URL

**Expected output**:
```
✅ Deployment Complete!
Your API is now available at:
  http://emotion-detection-api.eastus.azurecontainer.io:8000

API Documentation (Swagger UI):
  http://emotion-detection-api.eastus.azurecontainer.io:8000/docs
```

### Option B: Deploy to Azure App Service (Recommended for Production)

```bash
# Make the script executable
chmod +x deploy_azure_webapp.sh

# Run the deployment
./deploy_azure_webapp.sh
```

This script will:
1. Create an Azure Resource Group
2. Create an Azure Container Registry (ACR)
3. Build the image in Azure (using ACR build tasks)
4. Create an App Service Plan
5. Create and configure a Web App
6. Enable continuous deployment
7. Provide you with the HTTPS URL

**Expected output**:
```
✅ Deployment Complete!
Your API is now available at:
  https://emotion-detection-api-12345.azurewebsites.net

API Documentation (Swagger UI):
  https://emotion-detection-api-12345.azurewebsites.net/docs
```

### Manual Deployment (Alternative)

If you prefer manual control:

```bash
# 1. Login to Azure
az login

# 2. Create resource group
az group create --name emotion-detection-rg --location eastus

# 3. Create container registry
az acr create --resource-group emotion-detection-rg \
  --name emotiondetectionacr --sku Basic --admin-enabled true

# 4. Build image in ACR
az acr build --registry emotiondetectionacr \
  --image emotion-detection-api:latest \
  --file Dockerfile.api .

# 5. Create App Service Plan
az appservice plan create --name emotion-api-plan \
  --resource-group emotion-detection-rg \
  --is-linux --sku B2

# 6. Create Web App
az webapp create --resource-group emotion-detection-rg \
  --plan emotion-api-plan --name emotion-api-webapp \
  --deployment-container-image-name emotiondetectionacr.azurecr.io/emotion-detection-api:latest

# 7. Configure container settings
az webapp config container set \
  --name emotion-api-webapp \
  --resource-group emotion-detection-rg \
  --docker-custom-image-name emotiondetectionacr.azurecr.io/emotion-detection-api:latest \
  --docker-registry-server-url https://emotiondetectionacr.azurecr.io
```

---

## 🧪 Testing the API

### 1. Using the Test Script

```bash
# Test local deployment
python api/test_api.py http://localhost:8000

# Test Azure deployment
python api/test_api.py https://your-app-url.azurewebsites.net
```

### 2. Using cURL

```bash
# Health check
curl https://your-app-url/health

# Get API info
curl https://your-app-url/

# Predict emotion
curl -X POST "https://your-app-url/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

### 3. Using Python

```python
import requests

# API endpoint
url = "https://your-app-url/predict"

# Upload image
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    
print(response.json())
```

### 4. Using the Swagger UI

Navigate to `https://your-app-url/docs` for interactive API testing.

---

## 📊 Monitoring and Logs

### View Application Logs

**For App Service**:
```bash
# Stream logs in real-time
az webapp log tail --name emotion-api-webapp \
  --resource-group emotion-detection-rg

# Download logs
az webapp log download --name emotion-api-webapp \
  --resource-group emotion-detection-rg
```

**For Container Instances**:
```bash
# View logs
az container logs --resource-group emotion-detection-rg \
  --name emotion-api-aci

# Stream logs
az container attach --resource-group emotion-detection-rg \
  --name emotion-api-aci
```

### Azure Portal Monitoring

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your resource group
3. Click on your App Service or Container Instance
4. View metrics: CPU, Memory, HTTP requests, response times

### Set Up Application Insights (Optional)

```bash
# Create Application Insights
az monitor app-insights component create \
  --app emotion-api-insights \
  --location eastus \
  --resource-group emotion-detection-rg

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app emotion-api-insights \
  --resource-group emotion-detection-rg \
  --query instrumentationKey -o tsv)

# Add to Web App settings
az webapp config appsettings set \
  --name emotion-api-webapp \
  --resource-group emotion-detection-rg \
  --settings APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY
```

---

## 💰 Cost Optimization

### Estimated Monthly Costs

| Service | Tier | Cost (USD/month) |
|---------|------|------------------|
| Container Registry | Basic | ~$5 |
| Container Instances | 2 vCPU, 4GB | ~$30-50 |
| App Service | B2 (2 core, 3.5GB) | ~$55-75 |
| App Service | S1 (1 core, 1.75GB) | ~$70 |
| Bandwidth | First 100GB free | Variable |

### Cost Saving Tips

1. **Use App Service Free Tier for Development**
   ```bash
   az appservice plan create --name emotion-api-plan \
     --resource-group emotion-detection-rg \
     --sku FREE --is-linux
   ```

2. **Stop Resources When Not in Use**
   ```bash
   # Stop App Service
   az webapp stop --name emotion-api-webapp \
     --resource-group emotion-detection-rg
   
   # Delete Container Instance
   az container delete --name emotion-api-aci \
     --resource-group emotion-detection-rg
   ```

3. **Use Auto-scaling** (for production)
   ```bash
   az monitor autoscale create \
     --resource-group emotion-detection-rg \
     --resource emotion-api-webapp \
     --resource-type Microsoft.Web/sites \
     --min-count 1 --max-count 3 \
     --count 1
   ```

4. **Delete Unused Resources**
   ```bash
   # Delete entire resource group
   az group delete --name emotion-detection-rg --yes --no-wait
   ```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Container won't start

**Check logs**:
```bash
az webapp log tail --name emotion-api-webapp \
  --resource-group emotion-detection-rg
```

**Common causes**:
- Missing model files
- Insufficient memory (increase to 4GB+)
- Port configuration mismatch

#### 2. Model file not found

**Solution**: Ensure model files are in the `models/` directory before building:
```bash
ls -la models/
# Should show: emotion_enhanced_cnn_*.h5
```

#### 3. Out of memory errors

**Solution**: Increase container memory:
```bash
# For ACI
az container create --memory 8  # 8GB

# For App Service, upgrade to B3 or S tier
az appservice plan update --name emotion-api-plan \
  --resource-group emotion-detection-rg --sku B3
```

#### 4. Slow predictions

**Solutions**:
- Use GPU-enabled compute (Azure Container Instances GPU)
- Optimize model (quantization, pruning)
- Enable batch processing
- Add Redis cache for frequent requests

#### 5. 502 Bad Gateway

**Causes**:
- Container still starting (wait 2-3 minutes)
- Application crashed (check logs)
- Health check failing

**Solution**:
```bash
# Check container status
az webapp show --name emotion-api-webapp \
  --resource-group emotion-detection-rg \
  --query state

# Restart the app
az webapp restart --name emotion-api-webapp \
  --resource-group emotion-detection-rg
```

### Debug Mode

Enable debug logging in the FastAPI app by modifying `api/main.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🔄 Continuous Deployment

The Azure App Service deployment automatically sets up a webhook for continuous deployment. When you push a new image to ACR:

```bash
# Rebuild and push
az acr build --registry emotiondetectionacr \
  --image emotion-detection-api:latest \
  --file Dockerfile.api .

# App Service will automatically pull and deploy the new image
```

### GitHub Actions CI/CD (Optional)

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Login to Azure
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Build and push to ACR
      run: |
        az acr build --registry emotiondetectionacr \
          --image emotion-detection-api:${{ github.sha }} \
          --file Dockerfile.api .
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/)
- [Azure App Service](https://docs.microsoft.com/en-us/azure/app-service/)
- [Azure Container Registry](https://docs.microsoft.com/en-us/azure/container-registry/)
- [Docker Documentation](https://docs.docker.com/)

---

## 🎉 Next Steps

1. **Secure the API**: Add authentication (API keys, OAuth2)
2. **Custom Domain**: Configure your own domain name
3. **SSL/TLS**: Enable HTTPS with custom certificates
4. **Rate Limiting**: Protect against abuse
5. **Caching**: Add Redis for improved performance
6. **Monitoring**: Set up alerts and dashboards
7. **Backup**: Implement model versioning and backups

---

## 📞 Support

For issues or questions:
- Check the [troubleshooting section](#troubleshooting)
- Review Azure documentation
- Check application logs
- Open an issue in the repository

**Happy Deploying! 🚀**
