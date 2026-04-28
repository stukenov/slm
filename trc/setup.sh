#!/bin/bash
# SozKZ TRC Grant — One-time setup script
# Запусти это в терминале: bash ~/slm/trc/setup.sh

set -e
echo "=== SozKZ TRC Setup ==="

# 1. Install gcloud if not present
if ! command -v gcloud &>/dev/null; then
    echo "Installing Google Cloud SDK via brew..."
    brew install --cask google-cloud-sdk
    echo ""
    echo "✅ gcloud installed"
else
    echo "✅ gcloud already installed"
fi

# 2. Find and copy the key file
KEY_DST="$HOME/slm/trc/credentials/sozkz-tpu-key.json"
mkdir -p "$(dirname "$KEY_DST")"

if [ -f "$KEY_DST" ]; then
    echo "✅ Key file already in place"
elif [ -f "$HOME/Downloads/sozkz-tpu-key.json" ]; then
    cp "$HOME/Downloads/sozkz-tpu-key.json" "$KEY_DST"
    echo "✅ Key copied from Downloads"
else
    echo "❌ Key file not found in ~/Downloads/"
    echo "   Download it from Cloud Shell: cloudshell dl ~/sozkz-tpu-key.json"
    echo "   Then run this script again"
    exit 1
fi

# 3. Authenticate gcloud with service account
echo "Activating service account..."
gcloud auth activate-service-account \
    sozkz-tpu@sozkz-trc.iam.gserviceaccount.com \
    --key-file="$KEY_DST"

# 4. Set project
gcloud config set project sozkz-trc

# 5. Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$KEY_DST"

# Add to shell profile if not already there
SHELL_RC="$HOME/.zshrc"
[ -f "$HOME/.bashrc" ] && [ ! -f "$HOME/.zshrc" ] && SHELL_RC="$HOME/.bashrc"

if ! grep -q "GOOGLE_APPLICATION_CREDENTIALS" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# SozKZ TRC Grant" >> "$SHELL_RC"
    echo "export GOOGLE_APPLICATION_CREDENTIALS=\"\$HOME/slm/trc/credentials/sozkz-tpu-key.json\"" >> "$SHELL_RC"
    echo "export GOOGLE_CLOUD_PROJECT=\"sozkz-trc\"" >> "$SHELL_RC"
    echo "✅ Environment variables added to $SHELL_RC"
fi

# 6. Verify
echo ""
echo "=== Verification ==="
gcloud config get project
gcloud auth list --filter="account:sozkz-tpu@sozkz-trc.iam.gserviceaccount.com" --format="value(account,status)"
echo ""
echo "✅ Setup complete!"
echo ""
echo "Теперь можешь использовать:"
echo "  gcloud compute tpus tpu-vm list --zone=us-central2-b"
echo "  python3 ~/slm/trc/scripts/list_tpus.py"
echo "  python3 ~/slm/trc/scripts/create_tpu.py --test"
