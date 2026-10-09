#!/bin/bash

# The directory to scan for subdirectories
TARGET_DIR="$1"

# Check if the target directory is provided
if [ -z "$TARGET_DIR" ]; then
    echo "Usage: $0 [directory]"
    exit 1
fi

# Check if the directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory does not exist."
    exit 1
fi

# Function to get the current commit hash and remote URL
get_git_info() {
    local dir=$1
    cd "$dir" || return # Exit if cannot cd into directory
    if [ -d ".git" ]; then
        # Get the current commit hash
        local hash=$(git rev-parse HEAD 2> /dev/null)
        # Get the remote URL
        local remote=$(git config --get remote.origin.url 2> /dev/null)
        if [ -z "$remote" ]; then
            remote="null"
        fi
        # Print as a JSON object
        echo "{\"folder\": \"$dir\", \"commit\": \"$hash\", \"url\": \"$remote\"}"
    else
        echo "" > /dev/null # Print nothing if not a git repository
    fi
    cd - > /dev/null # Go back to the previous directory
}
export -f get_git_info

# Collect all JSON results into an array
json_array=()
while IFS= read -r line; do
    json_array+=("$line")
done < <(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'get_git_info "$0"' {} \;)

# Print the array as a JSON array
printf "[%s]\n" "$(IFS=,; echo "${json_array[*]}")"

