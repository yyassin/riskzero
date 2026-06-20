#!/bin/bash
# Post-create script: runs once after the devcontainer is first created.
# Sets up Powerlevel10k, installs ty, and syncs project dependencies via uv.
set -euo pipefail

echo "==> [post-create] Starting setup..."

# ── Powerlevel10k ─────────────────────────────────────────────────────────────
P10K_DIR="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k"
if [[ ! -d "$P10K_DIR" ]]; then
    echo "==> [post-create] Cloning Powerlevel10k..."
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "$P10K_DIR"
else
    echo "==> [post-create] Powerlevel10k already present, skipping clone."
fi

# Copy dotfiles (overwrites any defaults installed by common-utils feature)
echo "==> [post-create] Copying dotfiles..."
cp .devcontainer/dotfiles/.zshrc "$HOME/.zshrc"
cp .devcontainer/dotfiles/.p10k.zsh "$HOME/.p10k.zsh"

# ── Persist zsh history in the mounted volume ─────────────────────────────────
mkdir -p /commandhistory
touch /commandhistory/.zsh_history
echo 'export HISTFILE=/commandhistory/.zsh_history' >> "$HOME/.zshrc"

# ── ty (global uv tool) ───────────────────────────────────────────────────────
# ty may already be installed in the image; this is a no-op if so.
echo "==> [post-create] Ensuring ty is installed..."
uv tool install ty 2>/dev/null || uv tool upgrade ty

# ── Project dependencies ──────────────────────────────────────────────────────
echo "==> [post-create] Installing project dependencies via uv sync..."
uv sync

echo ""
echo "==> [post-create] Setup complete."
echo "    Python venv: $(pwd)/.venv"
echo "    uv:          $(uv --version)"
echo "    ty:          $(ty --version 2>/dev/null || echo 'see: uv tool list')"
