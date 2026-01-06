#!/bin/bash

# Check if a folder name is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder-name>"
fi

FOLDER=$1

# Check if the specified folder exists
if [ ! -d "$FOLDER" ]; then
    echo "Error: Directory '$FOLDER' does not exist."
fi

# Change the group of the folder and its contents to gaa50073
chgrp -R gaa50073 "$FOLDER"

# Set read, write, and execute permissions for the group gaa50073
chmod -R g+rwx "$FOLDER"

echo "Permissions set for group gaa50073 on all files and directories in '$FOLDER'"

