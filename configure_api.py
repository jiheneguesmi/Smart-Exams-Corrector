#!/usr/bin/env python3
"""
Configuration helper - Setup LLM API key easily
"""

import os
import sys
from pathlib import Path

def main():
    print("\n" + "="*80)
    print("🔧 LLM API KEY CONFIGURATION")
    print("="*80 + "\n")
    
    # Paths
    env_file = Path(".env")
    
    # Check if .env exists
    if env_file.exists():
        print("✅ .env file found!")
        with open(env_file) as f:
            content = f.read()
            if "hf_" in content or "VOTRE_CLE" not in content:
                print("📝 Your .env file appears to be configured.")
                print("\nCurrent configuration:")
                for line in content.split("\n"):
                    if line.startswith("LLM_API_KEY="):
                        print(f"   LLM_API_KEY = {line.split('=')[1][:10]}***")
                    elif line.startswith("#") or not line.strip():
                        continue
                    else:
                        print(f"   {line}")
                return
    
    print("📋 AVAILABLE PROVIDERS:\n")
    providers = {
        "1": {
            "name": "🤗 Hugging Face (RECOMMENDED)",
            "url": "https://huggingface.co/settings/tokens",
            "key": "hf_xxxxxxxxxxxxx",
            "free": "✅ Yes (free tier available)"
        },
        "2": {
            "name": "🚀 Together AI",
            "url": "https://www.together.ai/",
            "key": "xxxxxxxxxxxxx",
            "free": "✅ Yes ($5 free credits)"
        },
        "3": {
            "name": "🔄 Replicate",
            "url": "https://replicate.com/",
            "key": "xxxxxxxxxxxxx",
            "free": "✅ Yes (free tier available)"
        },
    }
    
    for num, provider in providers.items():
        print(f"\n{num}. {provider['name']}")
        print(f"   Website: {provider['url']}")
        print(f"   Free tier: {provider['free']}")
        print(f"   Key format: {provider['key']}")
    
    print("\n" + "-"*80)
    print("\n💡 QUICK SETUP:\n")
    print("1. Choose a provider above (Hugging Face recommended)")
    print("2. Go to the website and create an API key")
    print("3. Copy the key")
    print("4. Edit .env file and replace 'hf_VOTRE_CLE_API_ICI' with your key")
    print("5. Save and run: python batch_process_exams.py\n")
    
    # Prompt for API key
    print("-"*80)
    api_key = input("\n🔑 Enter your API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Read current .env
        if env_file.exists():
            with open(env_file) as f:
                content = f.read()
        else:
            content = "# LLM Configuration\n"
        
        # Update API key
        lines = content.split("\n")
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("LLM_API_KEY="):
                lines[i] = f"LLM_API_KEY={api_key}"
                updated = True
                break
        
        if not updated:
            lines.insert(1, f"LLM_API_KEY={api_key}")
        
        # Write back
        with open(env_file, "w") as f:
            f.write("\n".join(lines))
        
        print(f"\n✅ API key saved to .env!")
        print(f"   Key: {api_key[:10]}***")
        print("\n🚀 You can now run: python batch_process_exams.py\n")
    else:
        print("\n⚠️  No API key configured.")
        print("Please manually edit .env file or run this script again.\n")

if __name__ == "__main__":
    main()
