#!/bin/sh
for file in .github/workflows/ci-cd.yaml src/frontend/templates/cart.html; do
    echo "$file"
    dir=$(echo "$file" | awk -F'/' '{print $2}' | sort -u)
    echo "$dir"
    if [ -d "src/$dir" ]; then
        echo "services=$dir"
    fi
done