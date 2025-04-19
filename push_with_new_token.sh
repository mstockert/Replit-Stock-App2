#!/bin/bash

# Use the NEW_GITHUB_TOKEN from environment variables
if [ -z "$NEW_GITHUB_TOKEN" ]; then
  echo "Error: NEW_GITHUB_TOKEN environment variable is not set."
  exit 1
fi

# Set up git user information
git config --global user.email "replit_user@example.com" 
git config --global user.name "Replit User"

# Remove any existing remote (to avoid duplicate entries)
git remote remove origin 2>/dev/null || true

# Add the remote with token embedded in the URL
git remote add origin "https://x-access-token:${NEW_GITHUB_TOKEN}@github.com/mstockert/Replit-Stock-App2.git"

# Make sure .git directory exists
if [ ! -d .git ]; then
  git init
  echo "Git repository initialized"
fi

# Add all files
git add -A

# Commit changes
git commit -m "Add RSI and MACD technical indicators, improve multi-stock comparison" --allow-empty

# Force push to GitHub using the token (use with caution)
git push -f origin main

# Reset remote URL to the public one (without token) for security
git remote set-url origin https://github.com/mstockert/Replit-Stock-App2.git

echo "Code pushed successfully to GitHub repository!"