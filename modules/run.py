#!/usr/bin/env python3
"""
RAG Application Runner
Checks dependencies and starts the application
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    issues = []
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        issues.append("requirements.txt not found")
    
    # Check if sources directory exists
    if not Path("sources").exists():
        issues.append("sources/ directory not found")
    
    # Check if Python packages are installed
    try:
        import langchain_community
        import langchain_ollama
        import chromadb
        import gradio
        import langgraph
    except ImportError as e:
        issues.append(f"Missing Python package: {str(e).split()[-1]}")
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434", timeout=2)
        if response.status_code != 200:
            issues.append("Ollama server not responding")
    except:
        issues.append("Ollama server not accessible")
    
    return issues

def start_ollama_if_needed():
    """Start Ollama server if it's not running"""
    try:
        import requests
        requests.get("http://localhost:11434", timeout=2)
        print("✅ Ollama server is running")
        return True
    except:
        print("🔧 Starting Ollama server...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # Wait for server to start
            
            # Check again
            requests.get("http://localhost:11434", timeout=2)
            print("✅ Ollama server started")
            return True
        except:
            print("❌ Failed to start Ollama server")
            print("💡 Please run 'ollama serve' manually in another terminal")
            return False

def main():
    """Main runner function"""
    print("🚀 Starting RAG Application...")
    print("=" * 50)
    
    # Check requirements
    issues = check_requirements()
    
    if issues:
        print("❌ Setup issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Run setup first:")
        print("   python setup.py")
        return 1
    
    # Start Ollama if needed
    if not start_ollama_if_needed():
        print("⚠️  Continuing without Ollama (some features may not work)")
    
    # Start the application
    print("🌐 Starting web application...")
    try:
        from app import main as app_main
        return app_main()
    except ImportError:
        print("❌ app.py not found or has import errors")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return 1

if __name__ == "__main__":
    exit(main())