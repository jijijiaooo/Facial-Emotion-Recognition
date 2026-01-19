# Deployment Guide - Emotion Detection API on Render

## 📋 Prerequisites

- GitHub account
- Render account (sign up at https://render.com)
- Your code pushed to a GitHub repository

## 🚀 Deploy to Render

### Step 1: Push Your Code to GitHub

```bash
# If not already initialized
git init
git add .
git commit -m "Prepare for Render deployment"

# Create a new repository on GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### Step 2: Connect to Render

1. Go to https://render.com and sign in
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select your emotion detection repository

### Step 3: Configure Your Service

Render will auto-detect the `render.yaml` configuration, but verify:

- **Name**: `emotion-detection-api`
- **Environment**: `Python`
- **Build Command**: `pip install -r requirements_api.txt`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 2`
- **Plan**: Free

### Step 4: Deploy

1. Click "Create Web Service"
2. Render will start building your app (this takes 5-10 minutes)
3. Watch the logs for any errors
4. Once deployed, you'll get a URL like: `https://emotion-detection-api-xxxx.onrender.com`

## ⚠️ Important Notes

### Model Files

The model file (`emotion_enhanced_cnn_20251130_234141.h5`) needs to be included in your repository:

```bash
# Make sure the model is tracked
git add models/emotion_enhanced_cnn_20251130_234141.h5
git commit -m "Add model file"
git push
```

**Note**: GitHub has a 100MB file size limit. If your model is larger:
- Use Git LFS (Large File Storage)
- Or host the model externally (AWS S3, Google Drive) and download it during build

### Free Tier Limitations

Render's free tier:
- ✅ Spins down after 15 minutes of inactivity
- ⚠️ Takes 30-60 seconds to wake up on first request
- ⚠️ Limited to 512MB RAM
- ⚠️ Shared CPU

### Memory Optimization

If you encounter memory issues, update `render.yaml`:

```yaml
startCommand: "gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 1 --max-requests 1000 --max-requests-jitter 50"
```

## 🧪 Testing Your Deployed API

### Method 1: Using the Test Client (Recommended)

1. Open `test_client.html` in your browser
2. Update the API URL to your Render URL
3. Test all endpoints:
   - Click "Test /health" to check if API is running
   - Click "Test /emotions" to get emotion list
   - Upload an image and click "Predict Emotion"

### Method 2: Using curl

```bash
# Test health endpoint
curl https://your-app.onrender.com/health

# Test emotions list
curl https://your-app.onrender.com/emotions

# Test root endpoint
curl https://your-app.onrender.com/
```

### Method 3: Using Python

```python
import requests
import base64

API_URL = "https://your-app.onrender.com"

# Test health
response = requests.get(f"{API_URL}/health")
print(response.json())

# Test prediction
with open("test_image.jpg", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()
    
response = requests.post(
    f"{API_URL}/predict",
    json={"image": encoded}
)
print(response.json())
```

## 🔧 Troubleshooting

### Build Fails

**Issue**: `dlib` build fails
**Solution**: Already handled - dlib is commented out in `requirements_api.txt`. The app uses fallback mode.

**Issue**: Out of memory during build
**Solution**: 
```yaml
# In render.yaml, reduce dependencies or use lighter versions
tensorflow==2.16.1  # Already optimized
```

### App Crashes on Startup

Check the Render logs:
1. Go to your service dashboard
2. Click "Logs"
3. Look for errors in red

Common issues:
- Missing model file
- Import errors
- Memory issues

### Slow Response Times

First request after inactivity takes 30-60 seconds (cold start). Subsequent requests are fast.

To keep the service warm:
- Use a service like UptimeRobot to ping your API every 14 minutes
- Upgrade to a paid plan ($7/month) for always-on

## 📝 Environment Variables

If you need to add environment variables:

1. Go to Render Dashboard → Your Service → Environment
2. Add variables:
   - `DEBUG=false`
   - `MODEL_PATH=models/emotion_enhanced_cnn_20251130_234141.h5`

## 🔄 Updating Your Deployment

Every time you push to your GitHub repository, Render automatically redeploys:

```bash
# Make changes
git add .
git commit -m "Update API"
git push

# Render will auto-deploy (takes 2-5 minutes)
```

## 📊 Monitoring

### View Logs
```
Render Dashboard → Your Service → Logs
```

### Check Metrics
```
Render Dashboard → Your Service → Metrics
```

Shows:
- CPU usage
- Memory usage
- Request count
- Response times

## 💰 Cost Considerations

**Free Tier**: $0/month
- Perfect for testing and small projects
- Spins down after inactivity
- 750 hours/month free

**Starter Plan**: $7/month
- Always-on (no spin down)
- Better performance
- More resources

## 🎉 Success Checklist

- [ ] Code pushed to GitHub
- [ ] Service created on Render
- [ ] Deployment successful (no errors in logs)
- [ ] Health endpoint returns "healthy"
- [ ] Test client can connect to API
- [ ] Image prediction works
- [ ] API URL saved for future reference

## 📚 Resources

- Render Documentation: https://render.com/docs
- Flask Deployment Guide: https://render.com/docs/deploy-flask
- Troubleshooting: https://render.com/docs/troubleshooting-deploys

## 🆘 Getting Help

If you encounter issues:
1. Check Render logs first
2. Review this troubleshooting section
3. Check Render community forum
4. Contact Render support (they're very responsive!)

---

**Your API is now live! 🎊**

Share your API URL: `https://emotion-detection-api-xxxx.onrender.com`
