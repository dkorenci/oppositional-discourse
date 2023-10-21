#!/bin/bash

directory="$1"
folder_to_delete="per-file-data"

if [ -z "$directory" ]; then
    echo "Please provide a directory path as a command line parameter."
    exit 1
fi

# Navigate to the directory
cd "$directory" || exit 1

# Find and delete files without .json extension, ie
# leave only raw span annotation .json files, and hugginface datastet .json files
find . ! -name "*.json" -type f -delete

# remove gamma analysis folders from raw span annotation folders
find . -type d -name "$folder_to_delete" -exec rm -r {} +

echo "Deletion completed."
