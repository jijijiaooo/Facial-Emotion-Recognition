#!/bin/bash
# Script to upload the Facial Emotion Recognition project to GitHub

echo "🚀 Uploading Facial Emotion Recognition to GitHub"
echo "=================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files
echo "📦 Adding files to Git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "⚠️ No changes to commit"
else
    # Commit changes
    echo "💾 Committing changes..."
    git commit -m "Initial commit: Facial Emotion Recognition System

Features:
- Enhanced fear and disgust detection
- Multiple GUI options (simple, full-featured, Raspberry Pi)
- Comprehensive dataset analysis tools
- Raspberry Pi optimization
- Research-based improvements using Action Units
- Real-time emotion detection with confidence scoring
- Cross-platform compatibility

Project Structure:
- GUI applications with touch-friendly interface
- Core emotion recognition algorithms
- Model optimization for edge deployment
- Dataset enhancement and diagnostic tools
- Comprehensive documentation and setup guides"

    echo "✅ Changes committed"
fi

echo ""
echo "🔗 Next Steps:"
echo "=============="
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: facial-emotion-recognition"
echo "   - Description: Enhanced facial emotion recognition with fear/disgust optimization"
echo "   - Make it public (recommended) or private"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2. Add your GitHub repository as remote:"
echo "   git remote add origin https://github.com/YOURUSERNAME/facial-emotion-recognition.git"
echo ""
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "🎯 Alternative: Use GitHub CLI (if installed):"
echo "   gh repo create facial-emotion-recognition --public --source=. --remote=origin --push"
echo ""
echo "📋 Repository Features:"
echo "- ✅ Professional README with badges and documentation"
echo "- ✅ MIT License for open source compatibility"
echo "- ✅ Contributing guidelines for community involvement"
echo "- ✅ GitHub Actions for automated testing"
echo "- ✅ Proper .gitignore for Python projects"
echo "- ✅ Organized project structure"
echo ""
echo "🎉 Your project is ready for GitHub!"