"""
Helper script to easily configure and test your LLM API setup
"""

import os
import sys
import getpass
from pathlib import Path


def create_env_file(api_key):
    """Crée un fichier .env avec la clé API"""
    env_file = Path(".env")
    
    # Vérifie si le fichier existe déjà
    if env_file.exists():
        response = input("\n.env file already exists. Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return False
    
    # Écrit la clé dans le fichier
    with open(env_file, 'w') as f:
        f.write(f"LLM_API_KEY={api_key}\n")
    
    print(f"✅ Created .env file with API key")
    return True


def set_env_var(api_key):
    """Affiche les commandes pour définir la variable d'environnement"""
    print("\n" + "="*70)
    print("SET ENVIRONMENT VARIABLE")
    print("="*70)
    
    print("\n📋 Windows (PowerShell):")
    print(f'  $env:LLM_API_KEY = "{api_key}"')
    
    print("\n📋 Mac/Linux (Bash):")
    print(f'  export LLM_API_KEY="{api_key}"')
    
    print("\n📋 Or create .env file (easier):")
    response = input("Create .env file now? (y/n): ").strip().lower()
    if response == 'y':
        create_env_file(api_key)


def show_services():
    """Affiche les services LLM disponibles"""
    print("\n" + "="*70)
    print("AVAILABLE LLM SERVICES")
    print("="*70)
    
    services = {
        "1": {
            "name": "Hugging Face (RECOMMENDED)",
            "url": "https://huggingface.co/settings/tokens",
            "free": "✅ Yes (limited)",
            "setup": """
            1. Go to https://huggingface.co/settings/tokens
            2. Click "Create new token"
            3. Select "Read" access
            4. Copy the token (starts with 'hf_')
            """,
            "env": "HUGGINGFACE_API_KEY"
        },
        "2": {
            "name": "Together AI",
            "url": "https://api.together.xyz/",
            "free": "✅ Free credits included",
            "setup": """
            1. Go to https://api.together.xyz/
            2. Sign up
            3. Get your API key from dashboard
            """,
            "env": "TOGETHER_API_KEY"
        },
        "3": {
            "name": "Replicate",
            "url": "https://replicate.com/",
            "free": "✅ Limited free tier",
            "setup": """
            1. Go to https://replicate.com/
            2. Sign up
            3. Get your token from account settings
            """,
            "env": "REPLICATE_API_KEY"
        }
    }
    
    for num, service in services.items():
        print(f"\n{num}. {service['name']}")
        print(f"   URL: {service['url']}")
        print(f"   Free: {service['free']}")
        print(f"   Setup:{service['setup']}")


def main():
    print("\n" + "="*70)
    print("  LLM API CONFIGURATION HELPER")
    print("="*70)
    
    print("\nChoose your action:")
    print("  1. Get API key (show services)")
    print("  2. Configure environment variable")
    print("  3. Create .env file")
    print("  4. Test LLM configuration")
    print("  5. Run full setup")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        show_services()
    
    elif choice == "2":
        api_key = getpass.getpass("Enter your API key (hidden): ").strip()
        if api_key:
            set_env_var(api_key)
        else:
            print("No API key provided.")
    
    elif choice == "3":
        api_key = getpass.getpass("Enter your API key (hidden): ").strip()
        if api_key:
            create_env_file(api_key)
        else:
            print("No API key provided.")
    
    elif choice == "4":
        print("\nTesting LLM configuration...")
        os.system("python test_llm_config.py")
    
    elif choice == "5":
        print("\n" + "="*70)
        print("FULL SETUP")
        print("="*70)
        
        # Étape 1: Afficher les services
        show_services()
        
        # Étape 2: Obtenir la clé API
        api_key = getpass.getpass("\nEnter your API key (hidden): ").strip()
        
        if not api_key:
            print("❌ No API key provided. Exiting.")
            return
        
        # Étape 3: Créer le fichier .env
        print("\n" + "="*70)
        print("CREATING .env FILE")
        print("="*70)
        
        create_env_file(api_key)
        
        # Étape 4: Tester la configuration
        print("\n" + "="*70)
        print("TESTING CONFIGURATION")
        print("="*70)
        print("\nTesting LLM API connection...")
        os.system("python test_llm_config.py")
        
        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("  1. Prepare your exam data in data/ folder")
        print("  2. Run: python batch_process_exams.py")
    
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
