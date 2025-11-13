#!/bin/bash
#============================================================#
#  IBEX User Path Sanity Check (summary-first style)
#  Verifies /ibex/user/$USER exists, permissions (rwx),
#  and performs write/read/delete smoke tests.
#============================================================#

set -u  # keep -u, drop -e so we always reach summary

# -------- Config --------
IBEX_ROOT="/ibex/user"
ME="${USER:-$(whoami)}"
TARGET="${IBEX_ROOT}/${ME}"
TMPFILE="${TARGET}/.ibex_access_test_$$.tmp"

# -------- Pretty printing --------
ok()   { echo "[✓] $*"; }
fail() { echo "[✗] $*" >&2; }
info() { echo "→ $*"; }
sec()  { echo; echo "$1"; echo "${1//?/-}"; }

# -------- Flags --------
PATH_OK=1
PERM_OK=1
WRITE_OK=1
READ_OK=1
DELETE_OK=1

# -------- Step 1 --------
sec "[1/4] Validating user space path"
info "User: ${ME}"
info "Target path: ${TARGET}"

if [[ ! -d "${TARGET}" ]]; then
  fail "Path does not exist or is not a directory: ${TARGET}"
  PATH_OK=0
else
  ok "Directory exists"
fi

# -------- Step 2 --------
sec "[2/4] Checking permissions (rwx)"
if (( PATH_OK == 1 )); then
  if [[ ! -x "${TARGET}" ]]; then fail "No execute (search) permission on directory"; PERM_OK=0; fi
  if [[ ! -r "${TARGET}" ]]; then fail "No read permission on directory"; PERM_OK=0; fi
  if [[ ! -w "${TARGET}" ]]; then fail "No write permission on directory"; PERM_OK=0; fi
  (( PERM_OK == 1 )) && ok "Read/Write/Execute permissions look good"
else
  PERM_OK=0
  info "Skipping permission details because path is invalid."
fi

# -------- Step 3 --------
sec "[3/4] Write test"
if (( PATH_OK == 1 && PERM_OK == 1 )); then
  if ! echo "ibex access test $(date +%s)" > "${TMPFILE}" 2>/dev/null; then
    fail "Failed to create a file in ${TARGET}"
    WRITE_OK=0
  else
    ok "Write succeeded (${TMPFILE})"
  fi
else
  WRITE_OK=0
  info "Skipping write test due to path/permission failure."
fi

# -------- Step 4 --------
sec "[4/4] Read & delete tests"
if (( WRITE_OK == 1 )); then
  if ! grep -q "ibex access test" "${TMPFILE}" 2>/dev/null; then
    fail "Created file but could not read it back"
    READ_OK=0
  else
    ok "Read back test file successfully"
  fi

  if ! rm -f "${TMPFILE}" 2>/dev/null; then
    fail "Could not delete test file ${TMPFILE}"
    DELETE_OK=0
  else
    ok "Delete succeeded"
  fi
else
  READ_OK=0
  DELETE_OK=0
  info "Skipping read/delete because write failed."
fi

# -------- Summary --------
echo
echo "Summary"
echo "-------"
(( PATH_OK  == 1 )) && ok "Path exists: ${TARGET}" \
                    || fail "Path missing or not a directory"
(( PERM_OK  == 1 )) && ok "Permissions OK (rwx)" \
                    || fail "Insufficient permissions (need rwx)"
(( WRITE_OK == 1 )) && ok "Write test passed" \
                    || fail "Write test failed"
(( READ_OK  == 1 )) && ok "Read test passed" \
                    || fail "Read test failed"
(( DELETE_OK== 1 )) && ok "Delete test passed" \
                    || fail "Delete test failed"

echo
if (( PATH_OK==1 && PERM_OK==1 && WRITE_OK==1 && READ_OK==1 && DELETE_OK==1 )); then
  echo "/ibex/user/${ME} is accessible."
  # Optional filesystem info
  info "Filesystem info:"
  df -h "${TARGET}" | sed -n '1,2p'
  exit 0
else
  exit 1
fi
