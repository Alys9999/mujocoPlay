#!/usr/bin/env bash
set -euo pipefail

APPLY=0
SKIP_UPDATE=0
TARGET_CODENAME="${TARGET_CODENAME:-}"
APT_MIRROR="${APT_MIRROR:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/repair_ubuntu_apt_sources.sh [--apply] [--target-codename jammy] [--mirror URL] [--skip-update]

What it does:
  - Detects the Ubuntu codename from /etc/os-release unless --target-codename is provided
  - Rewrites /etc/apt/sources.list to match that codename
  - Backs up the previous apt source files before changing anything
  - Comments out mismatched Ubuntu archive entries in /etc/apt/sources.list.d/*.list

Defaults:
  - dry-run only; nothing is changed unless --apply is passed
  - mirror is reused from the first current deb line when possible

Examples:
  bash scripts/repair_ubuntu_apt_sources.sh
  bash scripts/repair_ubuntu_apt_sources.sh --apply
  bash scripts/repair_ubuntu_apt_sources.sh --apply --target-codename jammy
  APT_MIRROR=http://archive.ubuntu.com/ubuntu bash scripts/repair_ubuntu_apt_sources.sh --apply
EOF
}

log() {
  printf '[repair_ubuntu_apt_sources] %s\n' "$*"
}

die() {
  printf '[repair_ubuntu_apt_sources] ERROR: %s\n' "$*" >&2
  exit 1
}

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    die "This action needs root privileges. Re-run with sudo or as root."
  fi

  sudo "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      ;;
    --skip-update)
      SKIP_UPDATE=1
      ;;
    --target-codename)
      shift
      [[ $# -gt 0 ]] || die "--target-codename requires a value"
      TARGET_CODENAME="$1"
      ;;
    --mirror)
      shift
      [[ $# -gt 0 ]] || die "--mirror requires a value"
      APT_MIRROR="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

[[ -f /etc/os-release ]] || die "/etc/os-release not found"
# shellcheck disable=SC1091
source /etc/os-release

[[ "${ID:-}" == "ubuntu" ]] || die "This script only supports Ubuntu hosts. Detected ID='${ID:-unknown}'."

if [[ -z "$TARGET_CODENAME" ]]; then
  TARGET_CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
fi
[[ -n "$TARGET_CODENAME" ]] || die "Unable to determine the target Ubuntu codename from /etc/os-release."

if [[ -z "$APT_MIRROR" ]]; then
  APT_MIRROR="$(
    awk '
      $1 == "deb" && $2 ~ /^https?:\/\/.*ubuntu\/?$/ { print $2; exit }
      $1 == "deb" && $2 ~ /^https?:\/\/.*ubuntu\/$/ { print $2; exit }
    ' /etc/apt/sources.list 2>/dev/null
  )"
fi
if [[ -z "$APT_MIRROR" ]]; then
  APT_MIRROR="http://archive.ubuntu.com/ubuntu"
fi

if [[ ! "$APT_MIRROR" =~ /$ ]]; then
  APT_MIRROR="${APT_MIRROR}/"
fi

RENDERED_SOURCES="$(cat <<EOF
deb ${APT_MIRROR} ${TARGET_CODENAME} main restricted universe multiverse
deb ${APT_MIRROR} ${TARGET_CODENAME}-updates main restricted universe multiverse
deb ${APT_MIRROR} ${TARGET_CODENAME}-backports main restricted universe multiverse
deb ${APT_MIRROR} ${TARGET_CODENAME}-security main restricted universe multiverse
EOF
)"

preview() {
  log "Host reports Ubuntu ${VERSION_ID:-unknown} (${TARGET_CODENAME})"
  log "Mirror: ${APT_MIRROR}"
  echo
  echo "Current /etc/apt/sources.list:"
  sed -n '1,120p' /etc/apt/sources.list 2>/dev/null || true
  echo
  echo "Proposed /etc/apt/sources.list:"
  printf '%s\n' "$RENDERED_SOURCES"
  echo

  if compgen -G "/etc/apt/sources.list.d/*.list" >/dev/null; then
    echo "Potentially mismatched Ubuntu entries in /etc/apt/sources.list.d:"
    awk -v codename="$TARGET_CODENAME" '
      $1 == "deb" && $2 ~ /ubuntu/ && $3 !~ ("^" codename "(-|$)") {
        print FILENAME ":" NR ": " $0
      }
    ' /etc/apt/sources.list.d/*.list || true
    echo
  fi
}

disable_mismatched_ubuntu_lists() {
  local list_file tmp_file
  for list_file in /etc/apt/sources.list.d/*.list; do
    [[ -e "$list_file" ]] || continue
    tmp_file="$(mktemp)"
    awk -v codename="$TARGET_CODENAME" '
      $1 == "deb" && $2 ~ /ubuntu/ && $3 !~ ("^" codename "(-|$)") {
        print "# disabled-by-repair_ubuntu_apt_sources.sh: " $0
        next
      }
      { print }
    ' "$list_file" >"$tmp_file"
    install -m 0644 "$tmp_file" "$list_file"
    rm -f "$tmp_file"
  done
}

preview

if [[ "$APPLY" != "1" ]]; then
  log "Dry run only. Re-run with --apply to write the new apt sources."
  exit 0
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_dir="/etc/apt/source-repair-backups/${timestamp}"

log "Backing up current apt source files to ${backup_dir}"
as_root mkdir -p "$backup_dir/sources.list.d"
if [[ -f /etc/apt/sources.list ]]; then
  as_root cp -a /etc/apt/sources.list "$backup_dir/sources.list"
fi
if compgen -G "/etc/apt/sources.list.d/*" >/dev/null; then
  as_root cp -a /etc/apt/sources.list.d/. "$backup_dir/sources.list.d/"
fi

temp_sources="$(mktemp)"
printf '%s\n' "$RENDERED_SOURCES" >"$temp_sources"
log "Writing /etc/apt/sources.list for codename ${TARGET_CODENAME}"
as_root install -m 0644 "$temp_sources" /etc/apt/sources.list
rm -f "$temp_sources"

if compgen -G "/etc/apt/sources.list.d/*.list" >/dev/null; then
  log "Commenting out mismatched Ubuntu archive entries in /etc/apt/sources.list.d/*.list"
  as_root bash -lc "$(declare -f disable_mismatched_ubuntu_lists); TARGET_CODENAME='$TARGET_CODENAME'; disable_mismatched_ubuntu_lists"
fi

if compgen -G "/etc/apt/sources.list.d/*.sources" >/dev/null; then
  log "Deb822 .sources files were not modified automatically; review them if apt still points at the wrong release."
fi

if [[ "$SKIP_UPDATE" == "1" ]]; then
  log "Skipping apt-get update because --skip-update was requested."
  exit 0
fi

log "Running apt-get update"
as_root apt-get update
log "Apt sources now match ${TARGET_CODENAME}"
