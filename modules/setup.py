#!/usr/bin/env python3
"""
RAG Application Setup Script
Automatically installs dependencies and sets up the environment
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm

def run_command(command, description="", check=True, timeout=300):
    """Run a command with timeout and proper output handling"""
    print(f"🔧 {description or command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        print(f"   {'✅' if result.returncode == 0 else '❌'} {result.stdout.strip()}")
        if result.stderr:
            print(f"   ⚠️  {result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"   ❌ Command timed out after {timeout} seconds")
        return False
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e.stderr.strip() if e.stderr else str(e)}")
        return False

def install_python_dependencies():
    """Install Python dependencies with progress feedback"""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first with timeout
    if not run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip",
        timeout=60
    ):
        return False

    # Install requirements with feedback
    print("🔧 Installing requirements...")
    try:
        process = subprocess.Popen(
            f"{sys.executable} -m pip install -r requirements.txt",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"   {output.strip()}")
                
        returncode = process.poll()
        
        if returncode != 0:
            error = process.stderr.read()
            print(f"   ❌ Error: {error.strip()}")
            return False
            
        print("✅ Dependencies installed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install dependencies: {str(e)}")
        return False
    

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_ollama():
    """Check if Ollama is installed and running"""
    print("🔍 Checking Ollama...")
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running")
        print("\n💡 Please follow these steps:")
        print("1. Download and install Ollama from https://ollama.com")
        print("2. Start Ollama")
        print("3. Run this script again")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {str(e)}")
        return False
    
    return False

def install_ollama():
    """Install Ollama on Windows"""
    print("🤖 Installing Ollama...")
    
    # Define the download URL and installer path
    ollama_url = "https://ollama.com/download/windows"
    installer_path = os.path.join(os.getcwd(), "ollama-installer.exe")
    
    try:
        # Download the installer
        print("📥 Downloading Ollama installer...")
        response = requests.get(ollama_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(installer_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=4096):
                    f.write(data)
                    pbar.update(len(data))
        
        # Run the installer
        print("🔧 Running installer...")
        subprocess.run([installer_path, '/SILENT'], check=True)
        
        # Clean up
        os.remove(installer_path)
        
        print("✅ Ollama installed successfully")
        print("💡 Please restart your computer to complete installation")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install Ollama: {str(e)}")
        print("💡 Please download and install Ollama manually from: https://ollama.com")
        return False
    
def pull_models():
    """Pull required Ollama models"""
    models = ["llama3.1", "nomic-embed-text"]
    
    for model in models:
        print(f"📥 Pulling model: {model}")
        if not run_command(f"ollama pull {model}", f"Downloading {model}"):
            print(f"❌ Failed to pull {model}")
            return False
    
    return True

def check_package_conflicts():
    """Check for potential package conflicts before installation"""
    print("🔍 Checking package dependencies...")
    
    try:
        result = subprocess.run(
            f"{sys.executable} -m pip check",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("⚠️  Detected potential package conflicts:")
            print(result.stdout.strip())
            
            response = input("\n Would you like to:\n"
                           "1. Create a fresh virtual environment (recommended)\n"
                           "2. Try force reinstall\n"
                           "3. Cancel installation\n"
                           "Choose (1-3): ")
            
            if response == "1":
                return create_virtual_environment()
            elif response == "2":
                return force_reinstall_dependencies()
            else:
                return False
    except Exception as e:
        print(f"❌ Error checking dependencies: {str(e)}")
        return False
    
    return True

def create_virtual_environment():
    """Create a fresh virtual environment"""
    print("\n🔧 Creating virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("⚠️  Existing virtual environment found. Removing...")
        try:
            if platform.system().lower() == "windows":
                subprocess.run("rmdir /S /Q venv", shell=True, check=True)
            else:
                subprocess.run("rm -rf venv", shell=True, check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to remove existing virtual environment")
            return False
    
    try:
        # Create new virtual environment
        subprocess.run(f"{sys.executable} -m venv venv", shell=True, check=True)
        
        # Get the correct paths for Python and pip in the virtual environment
        if platform.system().lower() == "windows":
            python_path = str(venv_path / "Scripts" / "python.exe")
            pip_path = str(venv_path / "Scripts" / "pip.exe")
        else:
            python_path = str(venv_path / "bin" / "python")
            pip_path = str(venv_path / "bin" / "pip")
        
        print("📦 Installing dependencies in virtual environment...")
        
        # Install pip first
        subprocess.run(f'"{python_path}" -m ensurepip --upgrade', shell=True, check=True)
        subprocess.run(f'"{python_path}" -m pip install --upgrade pip', shell=True, check=True)
        
        # Install requirements one by one to handle dependencies better
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        # Install core dependencies first
        core_deps = ["pydantic>=2.7.4,<3.0.0", "langchain-core>=0.3.32,<0.4.0"]
        for dep in core_deps:
            subprocess.run(f'"{pip_path}" install {dep}', shell=True, check=True)
        
        # Then install the rest
        for req in requirements:
            if req not in core_deps:
                subprocess.run(f'"{pip_path}" install {req}', shell=True, check=True)
        
        print("✅ Virtual environment created and dependencies installed")
        print("\n💡 To activate the virtual environment:")
        if platform.system().lower() == "windows":
            print("   Run: .\\venv\\Scripts\\activate")
        else:
            print("   Run: source venv/bin/activate")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {str(e)}")
        return False
    

def force_reinstall_dependencies():
    """Force reinstall all dependencies"""
    print("\n🔧 Force reinstalling dependencies...")
    
    try:
        with open("requirements.txt", "r") as f:
            packages = [line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')]
        
        for package in packages:
            subprocess.run(f"{sys.executable} -m pip uninstall -y {package}", shell=True)
        
        subprocess.run(f"{sys.executable} -m pip install --no-deps -r requirements.txt", shell=True, check=True)
        subprocess.run(f"{sys.executable} -m pip install -r requirements.txt", shell=True, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to force reinstall dependencies: {str(e)}")
        return False


def install_python_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Use requirements.txt from root directory
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    if not requirements_path.exists():
        print("❌ requirements.txt not found in root directory")
        return False
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Try to install requirements
    result = subprocess.run(
        f"{sys.executable} -m pip install -r {requirements_path}",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"   ❌ Error: {result.stderr.strip()}")
        print("\n⚠️  Detected package conflicts!")
        
        response = input("\nWould you like to:\n"
                        "1. Create a fresh virtual environment (recommended)\n"
                        "2. Try force reinstall\n"
                        "3. Cancel installation\n"
                        "Choose (1-3): ")
        
        if response == "1":
            return create_virtual_environment()
        elif response == "2":
            return force_reinstall_dependencies()
        else:
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    # Get root directory
    root_dir = Path(__file__).parent.parent
    
    directories = {
        "sources": root_dir / "sources",
        "vector_db": root_dir / "chroma_db",
        "logs": root_dir / "logs"
    }
    
    for name, directory in directories.items():
        try:
            directory.mkdir(exist_ok=True, parents=True)
            print(f"   ✅ Created {name} directory: {directory.relative_to(root_dir)}")
        except Exception as e:
            print(f"   ❌ Failed to create {name} directory: {e}")
            return False
    
    return True

def create_sample_documents():
    """Create sample documents if sources directory is empty"""
    sources_dir = Path(__file__).parent.parent / "sources"
    
    if not any(sources_dir.glob("*.txt")) and not any(sources_dir.glob("*.pdf")):
        print("📝 Creating sample documents...")
        
        sample_content = """# Sample Document

This is a sample document for testing the RAG system.

## About This System
This RAG (Retrieval-Augmented Generation) system can answer questions based on the documents you provide.

## Features
- Document loading and processing
- Vector database storage
- Intelligent question answering
- Conversation memory
- Multiple file format support

## Usage
1. Add your documents to the sources/ directory
2. Run the application
3. Ask questions about your documents

Feel free to replace this file with your own content!
"""
        
        with open(sources_dir / "sample_document.txt", "w", encoding="utf-8") as f:
            f.write(sample_content)
        
        print("   ✅ Created sample_document.txt")
    
    return True

def main():
    """Main setup function"""
    print("🚀 RAG Application Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install Python dependencies
    if not install_python_dependencies():
        return 1
    
    # Check Ollama (but don't try to install)
    if not check_ollama():
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    print("\n✅ Setup completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())