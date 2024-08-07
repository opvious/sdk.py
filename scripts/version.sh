#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail
shopt -s nullglob

usage() {
	local cmd="${0##*/}"
	cat <<-EOF
		Manage package version

		All versions declared in \`pyproject.toml\` should be release candidates.
		This script can be used to extract the final version during releases.

		Usage:
		  $cmd check # Check version
		  $cmd show [-t] # Show version, optionally trimmed of suffix
	EOF
}

fail() { # MSG
	echo "$1" >&2 && exit 1
}

show_version() {
	local trim=0
	while getopts t o "$@"; do
		case "$o" in
			t) trim=1 ;;
			*) fail 'bad option'
		esac
	done

	local version="$(poetry version -s)"
	if ! [[ $version =~ (.+)rc.+ ]]; then
		fail "unexpected version: $version"
	fi

	if (( $trim == 0 )); then
		echo "$version"
	else
		echo "${BASH_REMATCH[1]}"
	fi
}

check_version() {
	show_version >/dev/null
}

main() {
	if (($# == 0)); then
		usage
		exit 3
	fi
	subcmd="$1"
	shift

	case $subcmd in
		check) check_version ;;
		show) show_version "$@" ;;
		*) fail 'invalid subcommand' ;;
	esac
}

main "$@"
