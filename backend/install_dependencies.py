import subprocess
import sys

def install_packages():
    """Install all required packages"""
    packages = [
        "flask",
        "flask-cors",
        "numpy",
        "scikit-learn",
        "pandas",
        "joblib",
        "librosa",
        "opencv-python",
        "tensorflow",
        "scipy",
        "soundfile"
    ]
    
    print("Installing required packages...")
    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("\n✅ All packages installed!")

if __name__ == "__main__":
    install_packages()