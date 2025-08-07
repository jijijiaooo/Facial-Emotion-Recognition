#!/usr/bin/env python3
"""
Fix Kaggle API Setup Issues
Helps resolve SSL warnings and missing credentials
"""

import os
import json
import subprocess
import sys
from pathlib import Path

class KaggleSetupFixer:
    def __init__(self):
        self.home_dir = Path.home()
        self.kaggle_dir = self.home_dir / '.kaggle'
        self.kaggle_json = self.kaggle_dir / 'kaggle.json'
    
    def check_ssl_issue(self):
        """Check and provide solutions for SSL issues"""
        print("🔍 Checking SSL Configuration...")
        print("-" * 40)
        
        try:
            import ssl
            print(f"SSL Version: {ssl.OPENSSL_VERSION}")
            
            if "LibreSSL" in ssl.OPENSSL_VERSION:
                print("⚠️ LibreSSL detected - this can cause urllib3 warnings")
                print("\n💡 Solutions:")
                print("1. Downgrade urllib3:")
                print("   pip install 'urllib3<2.0'")
                print("\n2. Or upgrade to newer Python/OpenSSL:")
                print("   brew install python@3.11")
                print("   # Then recreate your virtual environment")
                
                return False
            else:
                print("✅ SSL configuration looks good")
                return True
                
        except Exception as e:
            print(f"❌ SSL check failed: {e}")
            return False
    
    def fix_ssl_issue(self):
        """Apply SSL fix by downgrading urllib3"""
        print("\n🔧 Applying SSL Fix...")
        print("-" * 25)
        
        try:
            # Downgrade urllib3 to avoid LibreSSL issues
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "urllib3<2.0"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ urllib3 downgraded successfully")
                print("⚠️ Warning messages should be reduced now")
                return True
            else:
                print(f"❌ Failed to downgrade urllib3: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ SSL fix failed: {e}")
            return False
    
    def check_kaggle_credentials(self):
        """Check if Kaggle credentials are properly set up"""
        print("\n🔑 Checking Kaggle Credentials...")
        print("-" * 35)
        
        # Check if .kaggle directory exists
        if not self.kaggle_dir.exists():
            print("❌ ~/.kaggle directory not found")
            return False
        
        # Check if kaggle.json exists
        if not self.kaggle_json.exists():
            print("❌ ~/.kaggle/kaggle.json not found")
            return False
        
        # Check if kaggle.json has correct permissions
        stat_info = os.stat(self.kaggle_json)
        permissions = oct(stat_info.st_mode)[-3:]
        
        if permissions != '600':
            print(f"⚠️ kaggle.json permissions are {permissions}, should be 600")
            return False
        
        # Check if kaggle.json has required fields
        try:
            with open(self.kaggle_json, 'r') as f:
                config = json.load(f)
            
            required_fields = ['username', 'key']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                print(f"❌ Missing fields in kaggle.json: {missing_fields}")
                return False
            
            if not config['username'] or not config['key']:
                print("❌ Empty username or key in kaggle.json")
                return False
            
            print("✅ Kaggle credentials are properly configured")
            return True
            
        except json.JSONDecodeError:
            print("❌ kaggle.json is not valid JSON")
            return False
        except Exception as e:
            print(f"❌ Error reading kaggle.json: {e}")
            return False
    
    def setup_kaggle_credentials(self):
        """Interactive setup of Kaggle credentials"""
        print("\n🔧 Setting up Kaggle Credentials...")
        print("-" * 35)
        
        print("📋 Steps to get your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll down to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Download the kaggle.json file")
        print("5. Enter the credentials below:")
        
        # Get credentials from user
        username = input("\n👤 Enter your Kaggle username: ").strip()
        key = input("🔑 Enter your Kaggle API key: ").strip()
        
        if not username or not key:
            print("❌ Username and key cannot be empty")
            return False
        
        # Create .kaggle directory
        self.kaggle_dir.mkdir(exist_ok=True)
        
        # Create kaggle.json
        config = {
            "username": username,
            "key": key
        }
        
        try:
            with open(self.kaggle_json, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Set correct permissions
            os.chmod(self.kaggle_json, 0o600)
            
            print(f"✅ Kaggle credentials saved to {self.kaggle_json}")
            print("✅ Permissions set to 600")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save credentials: {e}")
            return False
    
    def test_kaggle_connection(self):
        """Test if Kaggle API is working"""
        print("\n🧪 Testing Kaggle Connection...")
        print("-" * 30)
        
        try:
            result = subprocess.run([
                "kaggle", "competitions", "list", "--page-size", "1"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Kaggle API connection successful!")
                return True
            else:
                print(f"❌ Kaggle API test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Kaggle API test timed out")
            return False
        except Exception as e:
            print(f"❌ Kaggle API test error: {e}")
            return False
    
    def create_alternative_download_script(self):
        """Create alternative download methods"""
        print("\n📝 Creating Alternative Download Methods...")
        print("-" * 45)
        
        # Method 1: Direct download script
        direct_download = '''#!/usr/bin/env python3
"""
Direct Dataset Download (Alternative to Kaggle API)
Downloads datasets using direct URLs when possible
"""

import requests
import zipfile
import os
from pathlib import Path

def download_file(url, filename):
    """Download file with progress"""
    print(f"📥 Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\\r  Progress: {percent:.1f}%", end="", flush=True)
        
        print(f"\\n✅ {filename} downloaded successfully")
        return True
        
    except Exception as e:
        print(f"\\n❌ Download failed: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"📂 Extracting {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def main():
    print("🔄 Alternative Dataset Download")
    print("=" * 35)
    
    # Create datasets directory
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    print("\\n💡 Manual Download Instructions:")
    print("Since Kaggle API isn't working, please:")
    print("\\n1. Go to these URLs in your browser:")
    print("   • https://www.kaggle.com/datasets/msambare/fer2013")
    print("   • https://www.kaggle.com/datasets/tom99763/affectnet-fer")
    print("   • https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer")
    print("\\n2. Click 'Download' button on each page")
    print("3. Move downloaded zip files to the 'datasets' folder")
    print("4. Run this script again to extract them")
    
    # Check for existing zip files
    zip_files = list(datasets_dir.glob("*.zip"))
    
    if zip_files:
        print(f"\\n📦 Found {len(zip_files)} zip files:")
        for zip_file in zip_files:
            print(f"  • {zip_file.name}")
            
            # Extract each zip file
            extract_dir = datasets_dir / zip_file.stem
            extract_dir.mkdir(exist_ok=True)
            extract_zip(zip_file, extract_dir)
        
        print("\\n✅ All datasets extracted!")
        print("💡 Next: Run python merge_datasets.py")
    else:
        print("\\n📥 No zip files found in datasets/ folder")
        print("Please download manually from the URLs above")

if __name__ == "__main__":
    main()
'''
        
        with open('alternative_download.py', 'w') as f:
            f.write(direct_download)
        
        print("✅ Created alternative_download.py")
        
        # Method 2: Manual instructions
        manual_instructions = '''
# MANUAL DATASET DOWNLOAD INSTRUCTIONS

## If Kaggle API isn't working, follow these steps:

### 1. Create Kaggle Account
- Go to https://www.kaggle.com
- Sign up/login to your account

### 2. Download Datasets Manually

#### FER-2013 Dataset:
1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Click "Download" button (requires login)
3. Save fer2013.zip to your datasets/ folder

#### AffectNet Dataset:
1. Go to: https://www.kaggle.com/datasets/tom99763/affectnet-fer
2. Click "Download" button
3. Save affectnet-fer.zip to your datasets/ folder

#### Emotion Detection Dataset:
1. Go to: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
2. Click "Download" button
3. Save emotion-detection-fer.zip to your datasets/ folder

### 3. Extract Files
Run: python alternative_download.py

### 4. Merge Datasets
Run: python merge_datasets.py

## Alternative Sources (No Kaggle Required):

### Academic Datasets:
1. JAFFE: https://zenodo.org/record/3451524
2. CK+: http://www.jeffcohn.net/Resources/ (requires academic request)

### Free Image Sources:
1. Unsplash: https://unsplash.com/s/photos/emotion
2. Pixabay: https://pixabay.com/images/search/facial%20expression/
3. Pexels: https://www.pexels.com/search/emotion/

Search terms: "fear expression", "disgust face", "scared person", "disgusted expression"
'''
        
        with open('MANUAL_DOWNLOAD.md', 'w') as f:
            f.write(manual_instructions)
        
        print("✅ Created MANUAL_DOWNLOAD.md")
    
    def run_complete_fix(self):
        """Run complete fix process"""
        print("🔧 KAGGLE SETUP FIXER")
        print("=" * 30)
        
        # Step 1: Fix SSL issue
        ssl_ok = self.check_ssl_issue()
        if not ssl_ok:
            if input("\n🔧 Apply SSL fix? (y/n): ").lower() == 'y':
                self.fix_ssl_issue()
        
        # Step 2: Check/setup credentials
        creds_ok = self.check_kaggle_credentials()
        if not creds_ok:
            if input("\n🔧 Setup Kaggle credentials? (y/n): ").lower() == 'y':
                self.setup_kaggle_credentials()
        
        # Step 3: Test connection
        if self.check_kaggle_credentials():
            self.test_kaggle_connection()
        
        # Step 4: Create alternatives
        self.create_alternative_download_script()
        
        print("\n🎯 SUMMARY:")
        print("=" * 15)
        print("✅ SSL fix applied (if needed)")
        print("✅ Kaggle credentials checked/setup")
        print("✅ Alternative download methods created")
        
        print("\n💡 NEXT STEPS:")
        print("1. Try: kaggle datasets download -d msambare/fer2013")
        print("2. If still fails: python alternative_download.py")
        print("3. Or follow MANUAL_DOWNLOAD.md instructions")

def main():
    fixer = KaggleSetupFixer()
    fixer.run_complete_fix()

if __name__ == "__main__":
    main()