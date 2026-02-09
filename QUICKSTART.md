# Azure Deployment Quick Start Guide

## 🎯 Quick Summary

This project now includes everything you need to deploy to Azure and connect to Android!

### What's Been Added:

1. **REST API** - Flask app in `api/app.py`
2. **Docker Support** - Dockerfile and docker-compose.yml
3. **Azure Templates** - ARM template in `azure/`
4. **Documentation**:
   - [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md) - Complete Azure deployment guide
   - [ANDROID_INTEGRATION.md](ANDROID_INTEGRATION.md) - Android app integration
5. **Test Scripts** - `test_api.py` and `test_api.sh`

---

## ⚡ Quick Start (3 Options)

### Option 1: Deploy to Azure App Service (Recommended)

**Time: ~15 minutes | Cost: Free tier or ~$13/month**

```bash
# 1. Login to Azure
az login

# 2. Create resource group
az group create --name rg-emotion-api --location eastus

# 3. Build and push Docker image
docker build -t emotion-api .
docker tag emotion-api YOUR_DOCKERHUB_USERNAME/emotion-api:latest
docker push YOUR_DOCKERHUB_USERNAME/emotion-api:latest

# 4. Deploy to Azure
az appservice plan create --name asp-emotion --resource-group rg-emotion-api --is-linux --sku B1
az webapp create --resource-group rg-emotion-api --plan asp-emotion --name emotion-api-YOURNAME --deployment-container-image-name YOUR_DOCKERHUB_USERNAME/emotion-api:latest

# 5. Get your URL
az webapp show --name emotion-api-YOURNAME --resource-group rg-emotion-api --query defaultHostName -o tsv
```

Your API will be at: `https://emotion-api-YOURNAME.azurewebsites.net`

### Option 2: Test Locally First

```bash
# 1. Install dependencies
pip install -r requirements_api.txt

# 2. Run API locally
python api/app.py

# 3. In another terminal, test it
python test_api.py http://localhost:8000
```

### Option 3: Use Docker Locally

```bash
# 1. Build and run with Docker
docker-compose up --build

# 2. Test
python test_api.py http://localhost:8000
```

---

## 📱 Connect to Android

See [ANDROID_INTEGRATION.md](ANDROID_INTEGRATION.md) for complete guide.

**Quick steps:**

1. Add dependencies to `build.gradle`:
   - Retrofit, Gson, CameraX
2. Update API URL in your Android code
3. Send base64-encoded images to `/api/detect`
4. Display results!

**Example API call from Android:**

```kotlin
// Retrofit service
interface EmotionApi {
    @POST("api/detect")
    suspend fun detect(@Body request: EmotionRequest): EmotionResponse
}

// Usage
val response = apiService.detect(EmotionRequest(base64Image))
println("Emotion: ${response.faces[0].emotion}")
```

---

## 📚 Documentation

- **[AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md)** - Step-by-step Azure deployment
- **[ANDROID_INTEGRATION.md](ANDROID_INTEGRATION.md)** - Complete Android integration
- **[README.md](README.md)** - Main project documentation

---

## 🧪 Testing

```bash
# Test local API
python test_api.py

# Test Azure API
python test_api.py https://your-app.azurewebsites.net

# Quick shell test
./test_api.sh https://your-app.azurewebsites.net
```

---

## 💰 Cost Estimates (Azure Student Credits)

| Service | Tier | Cost/Month | Notes |
|---------|------|------------|-------|
| App Service | F1 (Free) | $0 | Limited resources |
| App Service | B1 (Basic) | ~$13 | Recommended |
| Container Instances | Pay-per-use | ~$0.045/hour | Only when running |
| Container Registry | Basic | ~$5 | For storing images |

**Student tip:** Start with F1 or ACI to save credits!

---

## 🆘 Troubleshooting

### API returns 500 error
- Check logs: `az webapp log tail --name YOUR_APP --resource-group rg-emotion-api`
- Ensure model file exists in `models/` directory
- Check if dlib landmarks file is present

### Android can't connect
- Verify API URL is correct
- Check internet permission in AndroidManifest.xml
- Enable cleartext traffic for HTTP (or use HTTPS)
- Test API with browser first

### Slow response
- Resize images before sending (max 640x480)
- Reduce JPEG quality (60-80%)
- Use streaming endpoint for video

---

## 🚀 Next Steps

1. ✅ Deploy to Azure (follow AZURE_DEPLOYMENT.md)
2. ✅ Test with `test_api.py`
3. ✅ Create Android app (follow ANDROID_INTEGRATION.md)
4. ✅ Add authentication (API keys)
5. ✅ Monitor costs in Azure Portal
6. ✅ Optimize for production

---

## 📞 Support

- Azure Documentation: https://docs.microsoft.com/azure
- Azure Student: https://azure.microsoft.com/free/students
- Flask Documentation: https://flask.palletsprojects.com
- Android Developer: https://developer.android.com

Good luck with your deployment! 🎉
