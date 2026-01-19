#!/bin/bash
# Development script for running code quality checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running code quality checks...${NC}"
echo ""

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must be run from project root directory${NC}"
    exit 1
fi

# Run black formatter check
echo -e "${YELLOW}Checking code formatting with black...${NC}"
if uv run black --check .; then
    echo -e "${GREEN}✓ Code formatting is correct${NC}"
else
    echo -e "${RED}✗ Code formatting issues found${NC}"
    echo -e "${YELLOW}Run 'uv run black .' to fix formatting${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All quality checks passed!${NC}"
