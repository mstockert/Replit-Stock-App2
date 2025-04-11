#!/bin/bash

# Create a file for the token (this will not be committed)
read -p "Enter your GitHub token: " GITHUB_TOKEN
echo "https://x-access-token:$GITHUB_TOKEN@github.com/mstockert/Replit-Stock-App.git" > .git/token_remote

# Set up the remote using the file with token
git remote set-url origin "$(cat .git/token_remote)"

# Push to the repository
git push -u origin main

# Remove the token file for security
rm .git/token_remote

# Reset remote URL to the public one (without token)
git remote set-url origin https://github.com/mstockert/Replit-Stock-App.git

echo "Code pushed successfully to GitHub repository!"