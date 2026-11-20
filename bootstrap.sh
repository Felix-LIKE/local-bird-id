#!/usr/bin/env bash
#
# bootstrap.sh - one-command, distro-aware setup for the bird-ID pipeline.
#
# Does everything: clones the repo if needed, installs the few system packages
# (git, python3 + venv, exiftool), creates a private Python environment for the
# project's libraries, and puts `birdid` on your PATH. It never touches your
# system Python beyond what the venv module needs.
#
# Usage:
#   ./bootstrap.sh                set up from inside a checkout (recommended)
#   ./bootstrap.sh --dir DIR      clone the repo into DIR first, then set up
#   ./bootstrap.sh --no-system    skip the system-package install step
#   ./bootstrap.sh --initialize   also download the BioCLIP 2 model + taxonomy
#   ./bootstrap.sh --update       git pull --ff-only an existing checkout
#   ./bootstrap.sh --dry-run      print what it would do, without doing it
#   ./bootstrap.sh --help         this text
set -eu

REPO_URL="https://github.com/Felix-LIKE/local-bird-id.git"
VENV_DIR="pipeline/.venv"
PIP_DEPS="pybioclip numpy lxml requests pillow"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

do_system=1
do_initialize=0
do_update=0
dry_run=0
clone_dir=""

usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dir)        clone_dir="${2:?--dir needs an argument}"; shift 2 ;;
        --no-system)  do_system=0; shift ;;
        --initialize) do_initialize=1; shift ;;
        --update)     do_update=1; shift ;;
        --dry-run)    dry_run=1; shift ;;
        --help|-h)    usage ;;
        *) printf 'birdid: unknown option: %s\n' "$1" >&2; usage ;;
    esac
done

say() { printf '%s\n' "$*"; }
die() { printf 'birdid: ERROR: %s\n' "$*" >&2; exit 1; }

# run a command unless --dry-run (preserves quoting via "$@")
maybe() {
    printf '  %s\n' "$*"
    [ "$dry_run" -eq 0 ] || return 0
    "$@"
}

# --------------------------------------------------------------------------
# 1. where is the repo?
# --------------------------------------------------------------------------
if [ -f "pipeline/birdid.py" ]; then
    REPO="$PWD"
    say "Using the checkout at: $REPO"
else
    if [ -n "$clone_dir" ]; then
        REPO="$clone_dir"
    else
        REPO="$PWD/local-bird-id"
    fi
    # absolute path (mkdir so a deep --dir works even before it exists)
    REPO="$(mkdir -p "$REPO" && cd "$REPO" && pwd)"
    if [ ! -d "$REPO/.git" ]; then
        say "Cloning the repo into $REPO ..."
        maybe git clone "$REPO_URL" "$REPO"
        if [ "$dry_run" -eq 1 ]; then
            say ""
            say "Dry run from outside a checkout: the full plan needs the repo."
            say "Re-run from inside the checkout (or run without --dry-run)."
            exit 0
        fi
    fi
fi

[ -f "$REPO/pipeline/birdid.py" ] || die "pipeline/birdid.py not found in $REPO"
cd "$REPO"

if [ "$do_update" -eq 1 ] && [ -d ".git" ]; then
    say "Updating the checkout ..."
    maybe git pull --ff-only
fi

# --------------------------------------------------------------------------
# 2. system packages (distro-aware)
# --------------------------------------------------------------------------
PM=""
PM_LABEL=""
SYS_CMD=""
if command -v pacman >/dev/null 2>&1; then
    PM="pacman"; PM_LABEL="Arch-based (pacman)"
    SYS_CMD="sudo pacman -S --needed --noconfirm git python python-pip perl-image-exiftool"
elif command -v apt-get >/dev/null 2>&1; then
    PM="apt"; PM_LABEL="Debian/Ubuntu (apt)"
    SYS_CMD="sudo apt-get update && sudo apt-get install -y git python3 python3-venv python3-pip libimage-exiftool-perl"
elif command -v dnf >/dev/null 2>&1; then
    PM="dnf"; PM_LABEL="Fedora/RHEL (dnf)"
    SYS_CMD="sudo dnf install -y git python3 python3-pip python3-virtualenv perl-Image-ExifTool"
elif command -v zypper >/dev/null 2>&1; then
    PM="zypper"; PM_LABEL="openSUSE (zypper)"
    SYS_CMD="sudo zypper install -y git python3 python3-pip python3-virtualenv perl-Image-ExifTool"
elif command -v apk >/dev/null 2>&1; then
    PM="apk"; PM_LABEL="Alpine (apk)"
    SYS_CMD="sudo apk add --no-cache git python3 py3-pip py3-virtualenv exiftool"
elif [ "$(uname -s 2>/dev/null || true)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    PM="brew"; PM_LABEL="macOS (Homebrew)"
    SYS_CMD="brew install git python exiftool"
fi

if [ "$do_system" -eq 1 ]; then
    if [ -n "$PM" ]; then
        say ""
        say "Installing system packages ($PM_LABEL): git, python3 (+ venv/pip), exiftool"
        printf '  %s\n' "$SYS_CMD"
        if [ "$dry_run" -eq 0 ]; then
            # may prompt for your sudo password
            bash -c "$SYS_CMD"
        fi
    else
        say ""
        say "Could not detect your package manager."
        say "Please install manually: git, python3 with the venv module, and exiftool,"
        say "then re-run:  ./bootstrap.sh --no-system"
        [ "$dry_run" -eq 1 ] || exit 1
    fi
else
    say ""
    say "Skipping system-package install (--no-system)."
fi

# sanity: the tools we need to continue
command -v python3 >/dev/null 2>&1 || die "python3 not found (install it, then re-run)"
python3 -m venv --help >/dev/null 2>&1 || die "python3 has no 'venv' module (install python3-venv / python3-virtualenv, then re-run)"
if ! command -v exiftool >/dev/null 2>&1; then
    say "WARNING: exiftool not found - identification will fail until you install it."
fi

# --------------------------------------------------------------------------
# 3. private Python environment (plain venv - fully isolated)
# --------------------------------------------------------------------------
say ""
say "Creating the Python environment in $REPO/$VENV_DIR ..."
maybe python3 -m venv "$VENV_DIR"
maybe "$VENV_DIR/bin/pip" install --upgrade pip
maybe "$VENV_DIR/bin/pip" install torch torchvision --index-url "$TORCH_INDEX"
maybe "$VENV_DIR/bin/pip" install $PIP_DEPS

# --------------------------------------------------------------------------
# 4. put `birdid` on PATH
# --------------------------------------------------------------------------
BIN_DIR="$HOME/.local/bin"
say ""
say "Linking birdid into $BIN_DIR ..."
maybe mkdir -p "$BIN_DIR"
maybe ln -sfn "$REPO/pipeline/birdid" "$BIN_DIR/birdid"

if printf '%s' "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
    say "  $BIN_DIR is already on your PATH."
else
    RC_FILE="$HOME/.profile"
    case "${SHELL:-}" in
        *zsh)  RC_FILE="$HOME/.zshrc" ;;
        *bash) RC_FILE="$HOME/.bashrc" ;;
    esac
    if ! grep -qs "\.local/bin" "$RC_FILE" 2>/dev/null; then
        printf 'export PATH="$HOME/.local/bin:$PATH"\n' >> "$RC_FILE"
        say "  added '$HOME/.local/bin' to your PATH in $RC_FILE (reopen the terminal or 'source $RC_FILE')."
    else
        say "  $RC_FILE already mentions ~/.local/bin - good."
    fi
fi

# --------------------------------------------------------------------------
# 5. verify + next steps
# --------------------------------------------------------------------------
say ""
if [ "$dry_run" -eq 0 ]; then
    if "$BIN_DIR/birdid" --doctor >/dev/null 2>&1; then
        say "Setup OK - 'birdid --doctor' passes."
    else
        say "Setup finished, but 'birdid --doctor' reported problems (see above)."
    fi
else
    say "Dry run only - nothing was changed."
fi

say ""
say "Next steps:"
say "  birdid --doctor     check the install"
say "  birdid --list       see the built-in eBird data"
if [ "$do_initialize" -eq 1 ]; then
    say "  birdid --initialize  (running now: downloads BioCLIP 2 + taxonomy, ~4 GB, once)"
    [ "$dry_run" -eq 0 ] && "$BIN_DIR/birdid" --initialize
else
    say "  birdid --initialize  one-time AI-model download (~4 GB) when you are ready"
fi
say ""
say "To use a folder of photos:  birdid /path/to/photos"
