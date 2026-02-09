# 🚀 Quick Start - Deploy to Azure in 15 Minutes

Your Docker image is now ready! Follow these simple steps to deploy to Azure.

## ✅ Prerequisites Complete

- ✅ Docker image built successfully: `emotion-detection-api:latest`
- ✅ Tested locally and working
- ✅ Model file included in the container

## 📦 What Was Fixed

The build errors you encountered were resolved by:
1. **dlib compatibility issue** - Updated to version 19.24.6 and made it optional
2. **Model output handling** - Fixed to support both single and multiple outputs
3. **Optimized Dockerfile** - Better layer caching and error handling

## 🎯 Next Steps - Choose Your Deployment Method

### Option 1: Azure Portal (Visual Guide) 👨‍💻

Perfect if you prefer clicking through a UI interface.

**📖 Read the full guide**: `AZURE_PORTAL_DEPLOYMENT_STEPS.md`

**Quick summary:**
1. Login to [portal.azure.com](https://portal.azure.com)
2. Create Resource Group
3. Create Container Registry
4. Push your Docker image
5. Deploy to Container Instances or App Service
6. Access your API!

**Estimated time**: 20-25 minutes

---

### Option 2: Automated Script (Fastest) ⚡

Perfect if you're comfortable with command line.

#### For Simple Deployment (Testing/Development):

```bash
# Make sure you're logged into Azure
az login

# Run the automated deployment script
./deploy_azure_aci.sh
```

**This will:**
- Create all resources automatically
- Build and push Docker image to Azure
- Deploy to Azure Container Instances
- Give you a public URL

**Estimated time**: 10-15 minutes

#### For Production Deployment:

```bash
# Login to Azure
az login

# Run the production deployment script
./deploy_azure_webapp.sh
```

**This will:**
- Create all resources automatically
- Build image in Azure (faster)
- Deploy to App Service with HTTPS
- Enable auto-scaling
- Set up continuous deployment

**Estimated time**: 12-18 minutes

---

## 🧪 Test Your Local Docker Image First (Optional)

Before deploying to Azure, you can test locally:

```bash
# Run the container
docker run -d -p 8000:8000 --name emotion-api emotion-detection-api:latest

# Test the health endpoint
curl http://localhost:8000/health

# Test the API info
curl http://localhost:8000/

# Open Swagger UI in your browser
open http://localhost:8000/docs

# Stop when done
docker stop emotion-api && docker rm emotion-api
```

---

## 📋 Manual Deployment Steps (If You Want Full Control)

### Step 1: Login to Azure

```bash
az login
```

### Step 2: Set Your Variables

```bash
# Edit these with your preferences
export RESOURCE_GROUP="emotion-detection-rg"
export LOCATION="eastus"
export REGISTRY_NAME="yourname-emotion-acr"  # Must be globally unique
export IMAGE_NAME="emotion-detection-api"
export TAG="latest"
```

### Step 3: Create Resource Group

```bash
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

### Step 4: Create Container Registry

```bash
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $REGISTRY_NAME \
  --sku Basic \
  --admin-enabled true
```

### Step 5: Push Docker Image to ACR

```bash
# Login to ACR
az acr login --name $REGISTRY_NAME

# Tag your image
docker tag emotion-detection-api:latest \
  ${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${TAG}

# Push to ACR
docker push ${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${TAG}
```

### Step 6A: Deploy to Container Instances (Simpler)

```bash
# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query passwords[0].value -o tsv)

# Create container instance
az container create \
  --resource-group $RESOURCE_GROUP \
  --name emotion-api-aci \
  --image ${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${TAG} \
  --registry-login-server ${REGISTRY_NAME}.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --dns-name-label emotion-detection-api \
  --ports 8000 \
  --cpu 2 \
  --memory 4 \
  --environment-variables PORT=8000

# Get the URL
az container show \
  --resource-group $RESOURCE_GROUP \
  --name emotion-api-aci \
  --query ipAddress.fqdn -o tsv
```

### Step 6B: Deploy to App Service (Production)

```bash
# Create App Service Plan
az appservice plan create \
  --name emotion-api-plan \
  --resource-group $RESOURCE_GROUP \
  --is-linux \
  --sku B2

# Create Web App
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan emotion-api-plan \
  --name emotion-detection-webapp \
  --deployment-container-image-name ${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${TAG}

# Configure container settings
az webapp config container set \
  --name emotion-detection-webapp \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name ${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${TAG} \
  --docker-registry-server-url https://${REGISTRY_NAME}.azurecr.io

# Set environment variables
az webapp config appsettings set \
  --name emotion-detection-webapp \
  --resource-group $RESOURCE_GROUP \
  --settings WEBSITES_PORT=8000 PORT=8000 PYTHONUNBUFFERED=1

# Restart
az webapp restart \
  --name emotion-detection-webapp \
  --resource-group $RESOURCE_GROUP

# Get the URL
az webapp show \
  --name emotion-detection-webapp \
  --resource-group $RESOURCE_GROUP \
  --query defaultHostName -o tsv
```

---

## 🧪 Testing Your Deployed API

Once deployed, test your API:

```bash
# Replace with your actual URL
API_URL="https://your-app.azurewebsites.net"

# Test health
curl $API_URL/health

# Test API info
curl $API_URL/

# Run the test script
python api/test_api.py $API_URL

# Open Swagger UI
open $API_URL/docs
```

---

## 💰 Cost Estimates

| Component | Configuration | Monthly Cost |
|-----------|--------------|--------------|
| Container Registry | Basic | ~$5 |
| Container Instances | 2 vCPU, 4GB | ~$30-50 |
| App Service B2 | 2 cores, 3.5GB | ~$55-75 |
| Bandwidth | First 100GB free | Variable |

**Total for ACI**: ~$35-55/month
**Total for App Service**: ~$60-80/month

### 💡 Cost Saving Tips

1. **Stop when not using:**
   ```bash
   az container stop --name emotion-api-aci --resource-group $RESOURCE_GROUP
   az webapp stop --name emotion-detection-webapp --resource-group $RESOURCE_GROUP
   ```

2. **Delete when done testing:**
   ```bash
   az group delete --name $RESOURCE_GROUP --yes --no-wait
   ```

3. **Use Azure Free Credits:**
   - New Azure accounts get $200 free credits for 30 days
   - Students get $100/year with Azure for Students

---

## 📊 Monitoring Your Deployment

### View Logs

**Container Instances:**
```bash
az container logs --name emotion-api-aci --resource-group $RESOURCE_GROUP
```

**App Service:**
```bash
az webapp log tail --name emotion-detection-webapp --resource-group $RESOURCE_GROUP
```

### Check Status

**Container Instances:**
```bash
az container show --name emotion-api-aci --resource-group $RESOURCE_GROUP --query instanceView.state
```

**App Service:**
```bash
az webapp show --name emotion-detection-webapp --resource-group $RESOURCE_GROUP --query state
```

---

## 🔧 Troubleshooting

### Container won't start
- Check logs (see above)
- Ensure memory is at least 4GB
- Verify model file is in the Docker image

### Can't access the URL
- Wait 2-3 minutes for full startup
- Check if port 8000 is configured correctly
- Verify container is running

### Out of memory
- Increase to 8GB: `--memory 8`
- For App Service, upgrade to B3 tier

### Model not found
- Ensure `models/emotion_enhanced_cnn_*.h5` exists before building
- Rebuild the Docker image

---

## 🎉 What's Next?

After successful deployment:

1. **Test thoroughly** with real images
2. **Set up monitoring** with Application Insights
3. **Add authentication** for production use
4. **Configure custom domain** and SSL
5. **Set up CI/CD** with GitHub Actions
6. **Enable auto-scaling** for high traffic

---

## 📚 Additional Resources

- **Full Portal Guide**: `AZURE_PORTAL_DEPLOYMENT_STEPS.md`
- **Detailed Deployment Guide**: `AZURE_DEPLOYMENT_GUIDE.md`
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Azure Container Instances**: https://docs.microsoft.com/azure/container-instances/
- **Azure App Service**: https://docs.microsoft.com/azure/app-service/

---

## 🆘 Need Help?

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs
3. Verify all environment variables are set
4. Ensure the Docker image was built correctly
5. Check Azure service health

**Your Docker image is ready to deploy! Choose your method above and get started! 🚀**
