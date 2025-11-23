#!/bin/bash
#============================================================#
#  Tiny ImageNet Access Check (read-only)
#============================================================#

set -u  # keep -u (undefined vars error) but avoid -e so we reach summary

# -------- Config --------
TARGET="/ibex/reference/CV/tinyimagenet"

# -------- Pretty printing --------
ok()   { echo "[✓] $*"; }
fail() { echo "[✗] $*" >&2; }
info() { echo "→ $*"; }
sec()  { echo; echo "$1"; echo "${1//?/-}"; }

# -------- Flags (0/1) --------
PATH_OK=1
PERM_OK=1
LIST_OK=1
FILE_OK=1       # found a readable sample file
BYTES_OK=1      # could read bytes from that file

# -------- Step 1: Validate path --------
sec "[1/3] Validating dataset path"
info "Target: ${TARGET}"

if [[ ! -e "${TARGET}" ]]; then
  fail "Path does not exist: ${TARGET}"
  PATH_OK=0
fi
if [[ -e "${TARGET}" && ! -d "${TARGET}" ]]; then
  fail "Path exists but is not a directory: ${TARGET}"
  PATH_OK=0
fi
if (( PATH_OK == 1 )); then
  ok "Directory exists"
fi

# -------- Step 2: Permission checks (read-only expected) --------
sec "[2/3] Checking permissions"
if (( PATH_OK == 1 )); then
  # Need execute (search) and read on the directory
  if [[ ! -x "${TARGET}" ]]; then fail "Missing execute (search) permission on directory"; PERM_OK=0; fi
  if [[ ! -r "${TARGET}" ]]; then fail "Missing read permission on directory"; PERM_OK=0; fi
  if (( PERM_OK == 1 )); then ok "Directory permissions OK (read/list)"; fi
else
  PERM_OK=0
  info "Skipping permission details because path is invalid."
fi

# -------- Step 3: Readability smoke tests --------
sec "[3/3] Readability smoke test"

if (( PATH_OK == 1 && PERM_OK == 1 )); then
  # Try to list a few entries
  if ! ls -1 "${TARGET}" | head -n 10 >/dev/null 2>&1; then
    fail "Unable to list contents of ${TARGET}"
    LIST_OK=0
  else
    # Show a few entries for visibility (non-fatal if empty)
    ls -1 "${TARGET}" | head -n 10
  fi

  # Find a readable file
  SAMPLE_FILE="$(find "${TARGET}" -type f -readable -print -quit 2>/dev/null || true)"
  if [[ -z "${SAMPLE_FILE}" ]]; then
    fail "Could not find any readable files under ${TARGET}"
    FILE_OK=0
    BYTES_OK=0
  else
    info "Sample file: ${SAMPLE_FILE}"
    # Prove readability by counting bytes
    if ! BYTES=$(wc -c < "${SAMPLE_FILE}" 2>/dev/null); then
      fail "Found a file but could not read it: ${SAMPLE_FILE}"
      BYTES_OK=0
    else
      info "Readable bytes from sample file: ${BYTES}"
    fi
  fi
else
  LIST_OK=0
  FILE_OK=0
  BYTES_OK=0
  info "Skipping readability checks because path/permissions failed."
fi

if (( LIST_OK == 1 && FILE_OK == 1 && BYTES_OK == 1 )); then
  ok "At least one file is readable"
fi

# -------- Summary --------
echo
echo "Summary"
echo "-------"
if (( PATH_OK == 1 )); then
  ok "Path exists and is a directory"
else
  fail "Path missing or not a directory"
fi

if (( PERM_OK == 1 )); then
  ok "Permissions allow read/list"
else
  fail "Permission denied (missing read and/or execute)"
fi

if (( LIST_OK == 1 && FILE_OK == 1 && BYTES_OK == 1 )); then
  ok "Readability test passed (listed contents and read a file)"
else
  fail "Readability test failed"
fi

echo
if (( PATH_OK == 1 && PERM_OK == 1 && LIST_OK == 1 && FILE_OK == 1 && BYTES_OK == 1 )); then
  echo "Access to /ibex/reference/CV/tinyimagenet verified."
  exit 0
else
  exit 1
fi
