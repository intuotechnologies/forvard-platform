#!/bin/bash
# Script to run the test suite for the ForVARD Platform API

# Navigate to the api directory (if not already there)
cd "$(dirname "$0")"

# Source environment variables for testing
source setup_env.sh

# Make sure the test database file is not present
rm -f test.db || true

# Run the tests based on input arguments
if [ "$1" == "--unit" ]; then
    echo "Running unit tests..."
    poetry run pytest tests/unit -v
elif [ "$1" == "--integration" ]; then
    echo "Running integration tests..."
    poetry run pytest tests/integration -v
elif [ "$1" == "--coverage" ]; then
    echo "Running tests with coverage report..."
    poetry run pytest --cov=app --cov-report=term-missing tests/
else
    echo "Running all tests..."
    poetry run pytest tests/
fi

# Clean up after tests
rm -f test.db || true

echo "Tests completed." 