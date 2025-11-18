#!/bin/bash
# Remove temporary files from previous build locations if they exist
rm -rf ../public/_app 2>/dev/null || true
rm ../public/favicon.svg 2>/dev/null || true
rm ../public/index.html 2>/dev/null || true

# Also remove from current build location if needed
rm -rf ../dist/_app 2>/dev/null || true
rm ../dist/favicon.svg 2>/dev/null || true
rm ../dist/index.html 2>/dev/null || true
