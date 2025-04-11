#!/bin/bash

# Use the GITHUB_TOKEN from environment variables
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN environment variable is not set."
  exit 1
fi

# Set up git user information
git config --global user.email "replit_user@example.com" 
git config --global user.name "Replit User"

# Remove any existing remote (to avoid duplicate entries)
git remote remove origin

# Add the remote with token embedded in the URL (this won't display in logs)
git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/mstockert/Replit-Stock-App.git"

# Add all files
git add -A

# Commit changes
git commit -m "Updated Stock Market Dashboard code" --allow-empty

# Push to GitHub using the token
git push -f origin main

# Reset remote URL to the public one (without token) for security
git remote set-url origin https://github.com/mstockert/Replit-Stock-App.git

echo "Code pushed successfully to GitHub repository!"