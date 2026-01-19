#!/bin/bash
# Development script for formatting code

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Formatting code with black...${NC}"
echo ""

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must be run from project root directory${NC}"
    exit 1
fi

# Run black formatter
uv run black .

echo ""
echo -e "${GREEN}Code formatting complete!${NC}"
