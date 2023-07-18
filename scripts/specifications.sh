#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail
shopt -s nullglob

usage() {
  local cmd="${0##*/}"
  cat <<EOF
Manage specifications test resources

Usage:
  $cmd register # Register all test specifications
EOF
}

fail() { # MSG
  echo "$1" >&2 && exit 1
}

register_specifications() {
  for p in tests/notebooks/*; do
    poetry run python -m opvious register-notebook "$p"
  done
  for p in tests/sources/*; do
    poetry run python -m opvious register-sources "$p"
  done
}

main() {
  if (($# == 0)); then
    usage
    exit 3
  fi
  subcmd="$1"
  shift

  case $subcmd in
    register) register_specifications ;;
    *) fail 'invalid subcommand' ;;
  esac
}

main "$@"
