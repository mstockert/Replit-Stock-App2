#!/bin/bash

# Use the GITHUB_TOKEN from environment variables
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN environment variable is not set."
  exit 1
fi

# Create the repository using GitHub API
echo "Creating GitHub repository 'Replit-Stock-App'..."
curl -s -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user/repos \
  -d '{
    "name": "Replit-Stock-App",
    "description": "A Streamlit-based stock data visualization platform that provides comprehensive financial insights through interactive and user-friendly interfaces.",
    "private": true,
    "has_issues": true,
    "has_projects": true,
    "has_wiki": true
  }'

echo "Repository created. Now pushing code..."

# Set up git user information
git config --global user.email "replit_user@example.com" 
git config --global user.name "Replit User"

# Remove any existing remote (to avoid duplicate entries)
git remote remove origin

# Add the remote with token embedded in the URL
git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/mstockert/Replit-Stock-App.git"

# Make sure .git directory exists
if [ ! -d .git ]; then
  git init
  echo "Git repository initialized"
fi

# Add all files
git add -A

# Commit changes
git commit -m "Initial commit: Stock Market Dashboard" --allow-empty

# Push to GitHub using the token
git push -u origin main

# Reset remote URL to the public one (without token) for security
git remote set-url origin https://github.com/mstockert/Replit-Stock-App.git

echo "Code pushed successfully to GitHub repository!"