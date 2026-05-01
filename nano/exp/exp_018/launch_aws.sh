#!/bin/bash
set -euo pipefail

INSTANCE_TYPE="c7i.4xlarge"
AMI_ID="ami-0c7217cdde317cfec"
KEY_NAME="slm-tokenizer"
SSH_KEY="$HOME/.ssh/id_rsa"
SECURITY_GROUP="launch-wizard-1"
REGION="us-east-1"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=60 -i $SSH_KEY"

echo "=== exp_018: Morpheme BPE 256K Tokenizer ==="
echo "Instance: $INSTANCE_TYPE | Region: $REGION"
echo ""

# --- Step 1: Launch instance ---
echo "[1/5] Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-groups "$SECURITY_GROUP" \
    --region "$REGION" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=exp018-tokenizer},{Key=Project,Value=slm}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "  Instance ID: $INSTANCE_ID"

# --- Step 2: Wait for running ---
echo "[2/5] Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "  Public IP: $PUBLIC_IP"
echo "  Waiting 30s for SSH..."
sleep 30

# --- Step 3: Upload code + HF token ---
echo "[3/5] Uploading experiment code..."
REMOTE_DIR="/home/ubuntu/exp018"
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"

HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')}"
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set and ~/.cache/huggingface/token not found"
    exit 1
fi

ssh $SSH_OPTS ubuntu@"$PUBLIC_IP" "mkdir -p $REMOTE_DIR"

# Write HF token to temp file for SCP
HF_TMP=$(mktemp)
echo "$HF_TOKEN" > "$HF_TMP"

scp $SSH_OPTS \
    "$EXP_DIR/train_tokenizer.py" \
    "$EXP_DIR/morpheme_segmenter.py" \
    "$EXP_DIR/qazcorpora_model.py" \
    "$EXP_DIR/morpho_lemma_suf.pth" \
    "$EXP_DIR/setup_and_train.sh" \
    "$HF_TMP" \
    ubuntu@"$PUBLIC_IP":"$REMOTE_DIR/"

rm -f "$HF_TMP"
HF_TMP_NAME=$(basename "$HF_TMP")
ssh $SSH_OPTS ubuntu@"$PUBLIC_IP" "mv $REMOTE_DIR/$HF_TMP_NAME $REMOTE_DIR/.hf_token && chmod 600 $REMOTE_DIR/.hf_token"

echo "  Code + HF token uploaded."

# --- Step 4: Launch in screen ---
echo "[4/5] Launching training in screen..."
ssh $SSH_OPTS ubuntu@"$PUBLIC_IP" "chmod +x $REMOTE_DIR/setup_and_train.sh && screen -dmS exp018 bash -c '$REMOTE_DIR/setup_and_train.sh 2>&1 | tee $REMOTE_DIR/training.log'"

echo "  Screen session 'exp018' started."

# --- Step 5: Print monitoring commands ---
echo "[5/5] Done! Training running in background."
echo ""
echo "=== Monitor ==="
echo "  SSH:  ssh -i $SSH_KEY ubuntu@$PUBLIC_IP"
echo "  Log:  ssh -i $SSH_KEY ubuntu@$PUBLIC_IP 'tail -f $REMOTE_DIR/training.log'"
echo "  Screen: ssh -i $SSH_KEY -t ubuntu@$PUBLIC_IP 'screen -r exp018'"
echo ""
echo "=== After completion ==="
echo "  Terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Telegram notifications enabled — check bot for progress."
