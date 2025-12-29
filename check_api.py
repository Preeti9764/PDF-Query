"""
Quick diagnostic script to check if Gemini API is enabled and accessible
"""
import os
import sys
from dotenv import load_dotenv

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google-generativeai package not installed")
    print("Run: pip install google-generativeai")
    exit(1)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[X] No API key found in .env file")
    print("   Please add GOOGLE_API_KEY=your_key to your .env file")
    exit(1)

print(f"[OK] API key found: {api_key[:10]}...")

try:
    genai.configure(api_key=api_key)
    print("\n[CHECK] Checking available models...")
    
    models = genai.list_models()
    available_models = []
    
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            model_name = model.name.replace('models/', '')
            available_models.append(model_name)
            print(f"   [OK] {model_name}")
    
    if available_models:
        print(f"\n[SUCCESS] Found {len(available_models)} available chat model(s)")
        print(f"   Recommended model: {available_models[0]}")
        print("\n[OK] Your API is working! The app should work now.")
    else:
        print("\n[WARNING] No chat models found")
        print("   The Generative Language API might not be enabled")
        print("   Go to: https://console.cloud.google.com/apis/library")
        print("   Search for 'Generative Language API' and enable it")
        
except Exception as e:
    error_msg = str(e)
    print(f"\n[ERROR] {error_msg}")
    
    if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
        print("\n[FIX] Enable Generative Language API")
        print("   1. Go to: https://console.cloud.google.com/apis/library")
        print("   2. Search for 'Generative Language API'")
        print("   3. Click 'Enable'")
        print("   4. Wait 2-3 minutes, then try again")
    elif "NOT_FOUND" in error_msg or "404" in error_msg:
        print("\n[FIX] API might not be enabled or key is invalid")
        print("   1. Enable API: https://console.cloud.google.com/apis/library")
        print("   2. Check key permissions: https://console.cloud.google.com/apis/credentials")
        print("   3. Generate new key: https://makersuite.google.com/app/apikey")
    else:
        print("\n[FIX] Check your API key and permissions")

