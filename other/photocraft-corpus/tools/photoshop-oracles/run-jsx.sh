#!/bin/sh
# Run an ExtendScript file in the installed Photoshop via AppleScript `do javascript`.
# Usage: run-jsx.sh <script.jsx> [arg...]
# Arguments are passed to the script as `arguments[0..]`. Prints the script's result.
# Photoshop is addressed by bundle id, so any installed version works. The first run
# asks macOS for Automation permission (your terminal -> Adobe Photoshop).
set -eu
[ $# -ge 1 ] || { echo "usage: $0 <script.jsx> [arg...]" >&2; exit 2; }
script=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
shift
exec osascript - "$script" "$@" <<'APPLESCRIPT'
on run argv
	set src to read (POSIX file (item 1 of argv)) as «class utf8»
	set jsArgs to {}
	if (count of argv) > 1 then set jsArgs to items 2 thru -1 of argv
	with timeout of 3600 seconds
		tell application id "com.adobe.Photoshop"
			return do javascript src with arguments jsArgs
		end tell
	end timeout
end run
APPLESCRIPT
