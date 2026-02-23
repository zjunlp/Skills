#!/usr/bin/env bash
# install_skillnet.sh — Install the skillnet-ai Python SDK
# Targets: macOS / Linux (Ubuntu 24.04+, Debian 13+, Fedora 38+, etc.)
# Handles PEP 668 "externally-managed" environments automatically.
# Priority: uv > pipx > venv+pip > pip (--break-system-packages as last resort)
set -euo pipefail

PACKAGE="skillnet-ai"
VENV_DIR="${HOME}/.local/share/skillnet/venv"

echo "🧠 SkillNet SDK Installer"
echo "========================="

# ─── Already installed? ──────────────────────────────────────────────
if command -v skillnet &>/dev/null; then
  version=$(skillnet --version 2>/dev/null || echo "unknown")
  echo "✅ skillnet CLI already installed (${version})"
  exit 0
fi

# ─── Detect PEP 668 (externally-managed Python) ─────────────────────
is_externally_managed() {
  local py="${1:-python3}"
  local stdlib_dir
  stdlib_dir=$("$py" -c "import sysconfig; print(sysconfig.get_path('stdlib'))" 2>/dev/null) || return 1
  [ -f "${stdlib_dir}/EXTERNALLY-MANAGED" ]
}

# ─── Helper: try installing uv itself ────────────────────────────────
ensure_uv() {
  if command -v uv &>/dev/null; then return 0; fi

  echo "→ uv not found, attempting to install uv..."

  # Method 1: official installer (works on macOS + Linux, no root needed)
  if command -v curl &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null && {
      # shellcheck source=/dev/null
      export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
      command -v uv &>/dev/null && { echo "  ✅ uv installed via official installer"; return 0; }
    }
  fi

  # Method 2: brew (macOS)
  if command -v brew &>/dev/null; then
    brew install uv 2>/dev/null && { echo "  ✅ uv installed via Homebrew"; return 0; }
  fi

  # Method 3: apt (Debian/Ubuntu — available since Ubuntu 24.10+ repos)
  if command -v apt-get &>/dev/null && [ "$(id -u)" = "0" ] 2>/dev/null; then
    apt-get update -qq && apt-get install -y -qq uv 2>/dev/null && { echo "  ✅ uv installed via apt"; return 0; }
  fi

  return 1
}

# ─── Install strategy ────────────────────────────────────────────────

install_success=false

# Strategy 1: uv (preferred — fast, handles venvs transparently)
if ensure_uv; then
  echo "→ Installing ${PACKAGE} via uv..."
  uv pip install --system "${PACKAGE}" 2>/dev/null && install_success=true

  # If --system failed (PEP 668 / no permission), use uv with a venv
  if [ "$install_success" = false ]; then
    echo "  → System install blocked, creating venv at ${VENV_DIR}..."
    uv venv "${VENV_DIR}" 2>/dev/null
    VIRTUAL_ENV="${VENV_DIR}" uv pip install "${PACKAGE}" 2>/dev/null && install_success=true
  fi
fi

# Strategy 2: pipx (installs into isolated env, puts CLI on PATH)
if [ "$install_success" = false ] && command -v pipx &>/dev/null; then
  echo "→ Installing ${PACKAGE} via pipx..."
  pipx install "${PACKAGE}" 2>/dev/null && install_success=true
fi

# Strategy 3: Create a venv manually + pip inside it
if [ "$install_success" = false ]; then
  PY=""
  for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then PY="$candidate"; break; fi
  done

  if [ -n "${PY:-}" ]; then
    if is_externally_managed "$PY"; then
      echo "→ PEP 668 detected (externally-managed Python). Creating venv at ${VENV_DIR}..."
      "$PY" -m venv "${VENV_DIR}" 2>/dev/null && {
        "${VENV_DIR}/bin/pip" install --upgrade pip 2>/dev/null
        "${VENV_DIR}/bin/pip" install "${PACKAGE}" && install_success=true
      }
    else
      # Non-managed Python — direct pip is safe
      echo "→ Installing ${PACKAGE} via ${PY} -m pip..."
      "$PY" -m pip install "${PACKAGE}" 2>/dev/null && install_success=true
    fi
  fi
fi

# Strategy 4: last resort — --break-system-packages (warn user)
if [ "$install_success" = false ]; then
  PY=""
  for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then PY="$candidate"; break; fi
  done
  if [ -n "${PY:-}" ]; then
    echo "⚠️  Last resort: installing with --break-system-packages"
    echo "   This may affect system Python packages."
    "$PY" -m pip install --break-system-packages "${PACKAGE}" 2>/dev/null && install_success=true
  fi
fi

if [ "$install_success" = false ]; then
  echo "❌ Failed to install ${PACKAGE}."
  echo "   Please install Python 3.9+ and one of: uv, pipx, or pip."
  echo "   Recommended: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# ─── Ensure skillnet is on PATH ──────────────────────────────────────
export PATH="${HOME}/.local/bin:${VENV_DIR}/bin:${PATH}"

if command -v skillnet &>/dev/null; then
  echo "✅ skillnet CLI installed successfully."
  skillnet --help | head -5
else
  # Auto-create symlink for venv installs so skillnet is on PATH
  if [ -x "${VENV_DIR}/bin/skillnet" ]; then
    mkdir -p "${HOME}/.local/bin"
    ln -sf "${VENV_DIR}/bin/skillnet" "${HOME}/.local/bin/skillnet"
    export PATH="${HOME}/.local/bin:${PATH}"
    if command -v skillnet &>/dev/null; then
      echo "✅ skillnet CLI installed (symlinked to ~/.local/bin/skillnet)"
      skillnet --help | head -5
    else
      echo "✅ Installed and symlinked to ~/.local/bin/skillnet"
      echo "   Add to your shell profile if not on PATH:"
      echo "   export PATH=\"\$HOME/.local/bin:\${PATH}\""
    fi
  else
    echo "⚠️  Installation succeeded but 'skillnet' not found on PATH."
    echo "   Try: python3 -m skillnet_ai.cli --help"
  fi
fi
