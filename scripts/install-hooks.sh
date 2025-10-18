#!/bin/bash
# Install Git hooks for lattifai-python

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "📦 Installing Git hooks for lattifai-python..."

# Check if .git directory exists
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "❌ Error: .git directory not found. Are you in a Git repository?"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Create pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/sh
# Git pre-commit hook to run isort and ruff

echo "Running isort..."
isort src/ tests/ scripts/ --check-only --diff
ISORT_STATUS=$?

echo "Running ruff check..."
ruff check src/ tests/ scripts/
RUFF_CHECK_STATUS=$?

echo "Running ruff format check..."
ruff format --check src/ tests/ scripts/
RUFF_FORMAT_STATUS=$?

# If any check failed, exit with error
if [ $ISORT_STATUS -ne 0 ] || [ $RUFF_CHECK_STATUS -ne 0 ] || [ $RUFF_FORMAT_STATUS -ne 0 ]; then
    echo ""
    echo "❌ Pre-commit checks failed!"
    echo ""
    echo "To fix automatically, run:"
    echo "isort src/ tests/ scripts/"
    echo "ruff check --fix src/ tests/ scripts/"
    echo "ruff format src/ tests/ scripts/"
    echo ""
    echo "Or to bypass this check (not recommended), use: git commit --no-verify"
    exit 1
fi

echo "✅ All pre-commit checks passed!"
exit 0
EOF

# Make the hook executable
chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Git hooks installed successfully!"
echo ""
echo "The following hooks have been installed:"
echo "  - pre-commit: Runs isort and ruff checks before each commit"
echo ""
echo "To bypass hooks (not recommended), use: git commit --no-verify"
