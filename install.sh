## Bonus File ##
# install.sh - Quick installation script for Smart File Labeler

echo "📦 Installing Smart File Labeler..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$python_version < 3.8" | bc) -eq 1 ]]; then
    echo "❌ Python 3.8 or higher is required (found $python_version)"
    exit 1
fi

echo "✓ Python $python_version detected"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate and install
source venv/bin/activate
pip install --upgrade pip
pip install Pillow mutagen

echo "✅ Installation complete!"
echo ""
echo "Usage:"
echo "  source venv/bin/activate"
echo "  python file_labeler.py --help"
echo ""
echo "Quick test:"
echo "  python file_labeler.py --dry-run --verbose"
