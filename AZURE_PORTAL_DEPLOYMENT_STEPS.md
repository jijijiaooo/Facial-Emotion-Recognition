# Azure Portal Deployment - Step-by-Step Guide

Complete walkthrough for deploying the Facial Emotion Recognition API using the Azure Portal web interface.

---

## 📋 Prerequisites Checklist

Before you begin, ensure you have:

- ✅ Microsoft Azure account ([Sign up for free](https://azure.microsoft.com/free/))
- ✅ Docker Desktop installed and running
- ✅ Trained model file in `models/` directory (`emotion_enhanced_cnn_*.h5`)
- ✅ Azure CLI installed (for pushing Docker images)

---

## 🚀 Deployment Options

Choose one of these deployment methods:

### **Option A: Azure Container Instances (ACI)** - Recommended for beginners
- **Best for**: Quick deployment, testing, development
- **Cost**: ~$30-50/month
- **Setup time**: ~15 minutes
- **[Jump to ACI Instructions](#option-a-deploy-to-azure-container-instances-aci)**

### **Option B: Azure App Service (Web App for Containers)** - Recommended for production
- **Best for**: Production applications, auto-scaling
- **Cost**: ~$55-75/month
- **Setup time**: ~20 minutes
- **[Jump to App Service Instructions](#option-b-deploy-to-azure-app-service-production)**

---

## 🔧 Step 1: Prepare Your Docker Image

### 1.1 Build the Docker Image Locally

Open your terminal and navigate to your project directory:

```bash
cd /Users/jiaoshihlo/Codes/Facial-Emotion-Recognition-version-revised-dataset

# Build the Docker image
docker build -f Dockerfile.api -t emotion-detection-api:latest .
```

**Expected output:**
```
[+] Building 180.2s (15/15) FINISHED
 => [internal] load build definition from Dockerfile.api
 => => transferring dockerfile: 1.12kB
 ...
 => exporting to image
 => => exporting layers
 => => writing image sha256:abc123...
 => => naming to docker.io/library/emotion-detection-api:latest
```

### 1.2 Test Locally (Optional but Recommended)

```bash
# Run the container locally
docker run -p 8000:8000 emotion-detection-api:latest

# In another terminal, test it
curl http://localhost:8000/health
```

If you see `{"status":"healthy","detector_loaded":true}`, you're ready to deploy!

Press `Ctrl+C` to stop the local container.

---

## 🌐 Step 2: Login to Azure Portal

1. Open your web browser and go to: **https://portal.azure.com**
2. Sign in with your Microsoft account
3. You should see the Azure Portal dashboard

![Azure Portal Home](https://docs.microsoft.com/en-us/azure/media/index/portal.png)

---

## 📦 Step 3: Create a Resource Group

Resource groups are logical containers for Azure resources.

1. Click **"Resource groups"** in the left sidebar (or search for it in the top search bar)
2. Click **"+ Create"** button at the top
3. Fill in the details:
   - **Subscription**: Select your subscription
   - **Resource group name**: `emotion-detection-rg`
   - **Region**: Choose closest to you (e.g., `East US`, `West Europe`, `Southeast Asia`)
4. Click **"Review + create"**
5. Click **"Create"**

✅ **Resource group created!**

---

## 🐳 Step 4: Create Azure Container Registry (ACR)

ACR is where you'll store your Docker images.

### 4.1 Navigate to Container Registry

1. In the search bar at the top, type: **"Container registries"**
2. Click on **"Container registries"**
3. Click **"+ Create"** button

### 4.2 Configure the Registry

**Basics tab:**
- **Subscription**: Select your subscription
- **Resource group**: Select `emotion-detection-rg` (the one we just created)
- **Registry name**: Choose a **globally unique** name (must be lowercase, no spaces, 5-50 characters)
  - ⚠️ **IMPORTANT**: This name must be unique across ALL of Azure
  - **Try one of these formats**:
    - `yourname-emotion-acr` (replace `yourname` with your name)
    - `emotion-api-12345` (add random numbers)
    - `emotiondetect-yourcompany`
    - `fer-api-yourinitials-2026`
  - **Examples**: `john-emotion-acr`, `emotion-api-98765`, `fer-api-jdoe-2026`
  - **How to check if available**: Enter a name and Azure will show a ✅ or ❌ immediately
- **Location**: Same as your resource group
- **SKU**: Select **"Basic"** (cheapest option, ~$5/month)

> **💡 Tip**: Write down your chosen registry name - you'll need it for all the commands later!

**Networking tab:**
- Leave defaults (Public access)

**Encryption tab:**
- Leave defaults

### 4.3 Create the Registry

1. Click **"Review + create"**
2. Wait for validation to complete
3. Click **"Create"**
4. Wait 1-2 minutes for deployment to complete
5. Click **"Go to resource"**

✅ **Container Registry created!**

### 4.4 Enable Admin User (Important!)

1. In your Container Registry, click **"Access keys"** in the left menu (under Settings)
2. Toggle **"Admin user"** to **Enabled**
3. **IMPORTANT**: Note down these credentials (you'll need them later):
   - **Login server**: `[your-registry-name].azurecr.io`
   - **Username**: `[your-registry-name]`
   - **Password**: (click the copy icon next to password)
   
> 💾 **Save these values** - copy them to a text file on your desktop for easy reference!

---

## 📤 Step 5: Push Docker Image to ACR

Now we'll upload your Docker image to Azure.

### 5.1 Login to ACR via Command Line

Open your terminal:

```bash
# Login to Azure
az login (replace with YOUR registry name)
az acr login --name YOUR-REGISTRY-NAME
```

**Example:** If your registry is `john-emotion-acr`:
```bash
az acr login --name john-emotion-
# Login to your Container Registry
az acr login --name emotiondetectionacr
```

**Expected output:** `L (replace YOUR-REGISTRY-NAME with your actual registry name)
docker tag emotion-detection-api:latest YOUR-REGISTRY-NAME.azurecr.io/emotion-detection-api:latest
```

**Example:** If your registry is `john-emotion-acr`:
```bash
docker tag emotion-detection-api:latest john-emotion-
### 5.2 Tag Your Image

```bash
# Tag the image for ACR
docker tag emotion-detection-api:latest emotiondetectionacr.azurecr.io/emotion-detection-api:latest
``` (replace YOUR-REGISTRY-NAME)
docker push YOUR-REGISTRY-NAME.azurecr.io/emotion-detection-api:latest
```

**Example:** If your registry is `john-emotion-acr`:
```bash
docker push john-emotion-acr.azurecr.io/emotion-detection-api:latest
```

**This will take 5-10 minutes** depending on your internet speed.

**Expected output:**
```
The push refers to repository [your-registry-name
**This will take 5-10 minutes** depending on your internet speed.

**Expected output:**
```
The push refers to repository [emotiondesearch for your registry name)
3. Click **"Repositories"** in the left menu
4. You should see **`emotion-detection-api`** listed
5. Click on it to see the **`latest`** tag

✅ **Docker image uploaded to Azure!**
open test_emotion_api.html
> 🎯 **Remember**: Use YOUR actual registry name in all commands, not the examples!

### 5.4 Verify Upload

1. Go back to Azure Portal
2. Navigate to your Container Registry (`emotiondetectionacr`)
3. Click **"Repositories"** in the left menu
4. You should see **`emotion-detection-api`** listed
5. Click on it to see the **`latest`** tag

✅ **Docker image uploaded to Azure!**

---

## 🎯 Option A: Deploy to Azure Container Instances (ACI)

**Best for**: Simple deployment, testing, development

### A.1 Create Container Instance

1. In the search bar, type: **"Container instances"**
2. Click **"Container instances"**
3. Click **"+ Create"**

### A.2 Configure Container Instance

**Basics tab:**

- **Subscription**: Youyour registry (the one you created earlier)
- **Resource group**: `emotion-detection-rg`
- **Container name**: `emotion-api-aci`
- **Region**: Same as before
- **Availability zones**: None
- **SKU**: Standard
- **Image source**: Select **"Azure Container Registry"**
- **Registry**: Select `emotiondetectionacr`
- **Image**: Select `emotion-detection-api`
- **Image tag**: `latest`
- **OS type**: Linux
- **Size**: Click "Change size"
  - **Number of CPU cores**: 2
  - **Memory (GiB)**: 4
  - Click **"OK"**

### A.3 Configure Networking

Click **"Next: Networking"**

- **Networking type**: Public
- **DNS name label**: `emotion-detection-api` (or add your name/number if taken)
  - This will create: `emotion-detection-api.eastus.azurecontainer.io`
- **Domain name label scope**: Select **"Subscription"** (recommended)
  - This option is fine - it means your DNS name is unique within your subscription
  - Other options work too, but "Subscription" is a good default
- **Ports**: 
  - Protocol: TCP
  - Port: 8000

### A.4 Advanced Settings

Click **"Next: Advanced"**

- **Restart policy**: On failure
- **Environment variables**:
  - Click **"+ Add"**
    - Name: `PORT`
    - Value: `8000`
  - Click **"+ Add"**
    - Name: `PYTHONUNBUFFERED`
    - Value: `1`

### A.5 Create the Container

1. Click **"Review + create"**
2. Review your settings
3. Click **"Create"**
4. Wait 2-3 minutes for deployment

### A.6 Get Your URL

1. Click **"Go to resource"**
2. Look for **"FQDN"** (Fully Qualified Domain Name)
   - Example: `emotion-detection-api.eastus.azurecontainer.io`
3. Your API URL is: `http://[FQDN]:8000`

### A.7 Test Your Deployment

Open your browser and visit:
- **API Info**: `http://emotion-detection-api.eastus.azurecontainer.io:8000/`
- **Health Check**: `http://emotion-detection-api.eastus.azurecontainer.io:8000/health`
- **Swagger UI**: `http://emotion-detection-api.eastus.azurecontainer.io:8000/docs`

✅ **Deployment complete! Your API is live!**

[Skip to Testing Section](#-step-6-test-your-api)

---

## 🏢 Option B: Deploy to Azure App Service (Production)

**Best for**: Production applications, auto-scaling, better monitoring

### B.1 Create App Service Plan

1. In the search bar, type: **"App Service plans"**
2. Click **"App Service plans"**
3. Click **"+ Create"**

**Configure:**
- **Subscription**: Your subscription
- **Resource Group**: `emotion-detection-rg`
- **Name**: `emotion-api-plan`
- **Operating System**: Linux
- **Region**: Same as before
- **Pricing tier**: Click "Explore pricing plans"
  - Select **"Production"** tab
  - Choose **"B2"** (2 cores, 3.5 GB RAM, ~$55/month)
  - Or choose **"B1"** (1 core, 1.75 GB RAM, ~$13/month) for testing
  - Click **"Select"**

4. Click **"Review + create"**
5. Click **"Create"**

✅ **App Service Plan created!**

### B.2 Create Web App for Containers

1. In the search bar, type: **"App Services"**
2. Click **"App Services"**
3. Click **"+ Create"** → **"Web App"**

### B.3 Configure Web App

**Basics tab:**

- **Subscription**: Your subscription
- **Resource Group**: `emotion-detection-rg`
- **Name**: `emotion-detection-api-webapp` (must be globally unique)
  - *Try adding numbers or your name if taken*
  - This creates: `https://emotion-detection-api-webapp.azurewebsites.net`
- **Publish**: Docker Container
- **Operating System**: Linux
- **Region**: Same as before
- **App Service Plan**: Select `emotion-api-plan` (the one we created)
your registry (the one you created earlier)
### B.4 Configure Docker

Click **"Next: Docker"**

- **Options**: Single Container
- **Image Source**: Azure Container Registry
- **Registry**: Select `emotiondetectionacr`
- **Image**: Select `emotion-detection-api`
- **Tag**: `latest`
- **Startup Command**: Leave empty

### B.5 Review and Create

1. Click **"Review + create"**
2. Review your configuration
3. Click **"Create"**
4. Wait 2-3 minutes for deployment
5. Click **"Go to resource"**

### B.6 Configure Application Settings

In your Web App:

1. Click **"Configuration"** in the left menu (under Settings)
2. Click **"Application settings"** tab
3. Click **"+ New application setting"** and add:
   - **Name**: `WEBSITES_PORT` | **Value**: `8000` | Click "OK"
   - **Name**: `PORT` | **Value**: `8000` | Click "OK"
   - **Name**: `PYTHONUNBUFFERED` | **Value**: `1` | Click "OK"
4. Click **"Save"** at the top
5. Click **"Continue"** when prompted

### B.7 Restart the Web App

1. Click **"Overview"** in the left menu
2. Click **"Restart"** at the top
3. Click **"Yes"** to confirm
4. Wait 1-2 minutes for the app to restart

### B.8 Get Your URL

1. In the **Overview** page, look for **"Default domain"**
   - Example: `emotion-detection-api-webapp.azurewebsites.net`
2. Your API URL is: `https://[your-app-name].azurewebsites.net`

### B.9 Test Your Deployment

Open your browser and visit:
- **API Info**: `https://emotion-detection-api-webapp.azurewebsites.net/`
- **Health Check**: `https://emotion-detection-api-webapp.azurewebsites.net/health`
- **Swagger UI**: `https://emotion-detection-api-webapp.azurewebsites.net/docs`

✅ **Production deployment complete! Your API is live with HTTPS!**

---

## 🧪 Step 6: Test Your API

### 6.1 Using the Swagger UI (Easiest Method)

1. Open your browser to: `https://your-app-url/docs`
2. You'll see an interactive API documentation page
3. Click on **"POST /predict"** to expand it
4. Click **"Try it out"**
5. Click **"Choose File"** and select a photo with a face
6. Click **"Execute"**
7. Scroll down to see the JSON response with detected emotions!

**Example response:**
```json
{
  "success": true,
  "faces_detected": 1,
  "results": [
    {
      "face_id": 0,
      "emotion": "Happy",
      "confidence": 0.92,
      "bbox": {
        "x": 150,
        "y": 200,
        "width": 300,
        "height": 300
      }
    }
  ]
}
```

### 6.2 Using cURL (Command Line)

```bash
# Health check
curl https://your-app-url/health

# Get API information
curl https://your-app-url/

# Predict emotion from image
curl -X POST "https://your-app-url/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

### 6.3 Using Python

```python
import requests

# Your API URL
API_URL = "https://emotion-detection-api-webapp.azurewebsites.net"

# Test health
response = requests.get(f"{API_URL}/health")
print(response.json())

# Predict emotion
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{API_URL}/predict", files=files)
    print(response.json())
```

### 6.4 Using the Test Script

```bash
# From your project directory
python api/test_api.py https://your-app-url
```

---

## 📊 Step 7: Monitor Your Application

### 7.1 View Application Logs

**For Container Instances:**
1. Go to your Container Instance in Azure Portal
2. Click **"Containers"** in the left menu
3. Click **"Logs"** tab
4. View real-time logs

**For App Service:**
1. Go to your Web App in Azure Portal
2. Click **"Log stream"** in the left menu (under Monitoring)
3. View real-time logs

### 7.2 View Metrics

1. In your resource, click **"Metrics"** in the left menu
2. Add metrics to monitor:
   - CPU usage
   - Memory usage
   - HTTP requests
   - Response time

### 7.3 Set Up Alerts (Optional)

1. Click **"Alerts"** in the left menu
2. Click **"+ Create"** → **"Alert rule"**
3. Set conditions (e.g., CPU > 80%)
4. Configure actions (e.g., send email)

---

## 💰 Step 8: Manage Costs

### 8.1 Check Current Costs

1. In Azure Portal, search for **"Cost Management + Billing"**
2. Click **"Cost analysis"**
3. Filter by resource group: `emotion-detection-rg`
4. View your spending

### 8.2 Stop Resources When Not in Use

**Stop Container Instance:**
1. Go to your Container Instance
2. Click **"Stop"** at the top
3. Click **"Start"** when you need it again

**Stop App Service:**
1. Go to your Web App
2. Click **"Stop"** at the top
3. Click **"Start"** when you need it again

**Note:** You're still charged for the App Service Plan when stopped. To fully stop billing, delete the resources.

### 8.3 Delete Resources (When Done Testing)

**Delete entire resource group** (removes everything):
1. Go to **"Resource groups"**
2. Click on `emotion-detection-rg`
3. Click **"Delete resource group"**
4. Type the resource group name to confirm
5. Click **"Delete"**

---

## 🔧 Troub"Registry name not available" when creating ACR

**Problem**: The registry name you chose is already taken by someone else globally.

**Solution**:
1. **Try a more unique name** using one of these patterns:
   - Add your initials: `fer-api-jd-2026`
   - Add random numbers: `emotion-api-87654`
   - Add company name: `emotion-yourcompany-acr`
   - Add location: `emotion-westus-123`

2. **Use this command to check availability** before trying in the portal:
   ```bash
   az acr check-name --name YOUR-DESIRED-NAME
   ```
   
   This will tell you if the name is available!

3. **Naming requirements**:
   - 5-50 characters
   - Only lowercase letters and numbers
   - No spaces, hyphens in the middle are OK
   - Must start with a letter

**Example of checking availability**:
```bash
# Check if name is available
az acr check-name --name john-emotion-acr

# If available, you'll see:
# {
#   "nameAvailable": true
# }
```

### Issue: leshooting

### Issue: Container won't start

**Check logs:**
1. Navigate to your Container Instance or App Service
2. View logs (see Step 7.1)
3. Look for error messages

**Common fixes:**
- Increase memory to 4GB or more
- Check that PORT environment variable is set to 8000
- Verify the model file exists in the Docker image

### Issue: "Application Error" or 502 Bad Gateway

**Wait:** Container may still be starting (takes 2-3 minutes)

**Fix:**
1. Restart the application
2. Check logs for errors
3. Verify WEBSITES_PORT is set to 8000 (App Service)

### Issue: Can't access the URL

**Check:**
1. URL format is correct (http:// vs https://)
2. Port 8000 is included in URL (for ACI)
3. Container is running (status should be "Running")

### Issue: Out of memory errors

**Solution:**
1. Increase container memory to 8GB
2. For App Service, upgrade to B3 or S1 tier

### Issue: Slow predictions

**Solutions:**
- Upgrade to higher tier (more CPU/RAM)
- Optimize model (smaller size)
- Enable caching for frequent requests

---

## 🎉 Next Steps

Now that your API is deployed, you can:

### 1. Secure Your API
- Add API key authentication
- Configure CORS properly
- Set up rate limiting

### 2. Add Custom Domain
1. Buy a domain (e.g., from GoDaddy)
2. In App Service, go to **"Custom domains"**
3. Add your domain
4. Configure SSL certificate (free with Let's Encrypt)

### 3. Enable HTTPS for[YOUR-UNIQUE-REGISTRY-NAME]` ← **You chose this!**
- Use Azure Application Gateway
- Or deploy to App Service (has built-in HTTPS)

### 4. Set Up Continuous Deployment
- Connect to GitHub
- Auto-deploy when you push code changes

### 5. Scale Your Application
**For App Service:**
1. Go to **"Scale up (App Service plan)"** to get more power
2. Go to **"Scale out (App Service plan)"** to add more instances
3. Enable auto-scaling based on CPU usage

### 6. Add Monitoring
- Check if a registry name is available
az acr check-name --name YOUR-DESIRED-NAME

# Login to Container Registry (replace with YOUR name)
az acr login --name YOUR-REGISTRY-NAME

# Tag image (replace with YOUR registry name)
docker tag emotion-detection-api:latest YOUR-REGISTRY-NAME.azurecr.io/emotion-detection-api:latest

# Push image (replace with YOUR registry name)
docker push YOUR-REGISTRY-NAME.azurecr.io/emotion-detection-api:latest

# Test API
python api/test_api.py https://your-app-url
```

**💡 Remember to replace `YOUR-REGISTRY-NAME` with your actual registry name!**Azure Portal](https://portal.azure.com)
- [Azure Container Instances Docs](https://docs.microsoft.com/en-us/azure/container-instances/)
- [Azure App Service Docs](https://docs.microsoft.com/en-us/azure/app-service/)
- [Azure Container Registry Docs](https://docs.microsoft.com/en-us/azure/container-registry/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 💡 Quick Reference

### Your Resource Names
- Resource Group: `emotion-detection-rg`
- Container Registry: `emotiondetectionacr`
- Container Instance: `emotion-api-aci` (Option A)
- App Service Plan: `emotion-api-plan` (Option B)
- Web App: `emotion-detection-api-webapp` (Option B)

### Your URLs
- **Container Instance**: `http://[fqdn]:8000`
- **App Service**: `https://[app-name].azurewebsites.net`
- **Swagger UI**: Add `/docs` to your URL

### Important Commands
```bash
# Login to Azure
az login

# Login to Container Registry
az acr login --name emotiondetectionacr

# Push image
docker push emotiondetectionacr.azurecr.io/emotion-detection-api:latest

# Test API
python api/test_api.py https://your-app-url
```

---

## 🆘 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review application logs in Azure Portal
3. Verify all environment variables are set correctly
4. Ensure the Docker image was built correctly
5. Check Azure service health status

**Common Support Resources:**
- Azure Documentation
- Stack Overflow (tag: azure)
- Azure Support (paid plans)

---

**Congratulations! 🎉 You've successfully deployed your Emotion Detection API to Azure!**

Your API is now accessible globally and ready to detect emotions from facial images!
