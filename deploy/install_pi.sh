#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run with sudo: sudo ./deploy/install_pi.sh"
  exit 1
fi

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
INSTALL_DIR=/opt/gesture-bridge
SERVICE_USER=${GESTURE_BRIDGE_USER:-${SUDO_USER:-pi}}
if [[ "$SERVICE_USER" == "root" ]]; then
  echo "Set the desktop user: sudo GESTURE_BRIDGE_USER=<username> ./deploy/install_pi.sh"
  exit 1
fi
PYTHON_BIN=${GESTURE_BRIDGE_PYTHON:-/home/$SERVICE_USER/.pyenv/versions/3.12.8/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=$(command -v python3)
fi

install -d -o "$SERVICE_USER" -g "$SERVICE_USER" "$INSTALL_DIR"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.idea/' \
  --exclude 'venv/' \
  --exclude 'data/' \
  --exclude '.sequence_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.log' \
  "$SOURCE_DIR"/ "$INSTALL_DIR"/
"$PYTHON_BIN" -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" gpiozero
chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR"

if [[ ! -f "$INSTALL_DIR/gesture_recognizer.task" ]]; then
  echo "Missing gesture_recognizer.task; installation cannot continue."
  exit 1
fi

install -m 0644 "$SOURCE_DIR/deploy/gesture-bridge.service" /etc/systemd/system/gesture-bridge.service
sed -i "s/^User=.*/User=$SERVICE_USER/" /etc/systemd/system/gesture-bridge.service
if [[ ! -f /etc/gesture-bridge.env ]]; then
  install -m 0600 "$SOURCE_DIR/config.example.env" /etc/gesture-bridge.env
fi
systemctl daemon-reload
systemctl enable gesture-bridge.service

echo "Edit /etc/gesture-bridge.env, then run: sudo systemctl start gesture-bridge"
