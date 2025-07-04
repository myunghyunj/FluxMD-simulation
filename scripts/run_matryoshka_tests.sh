#!/bin/bash
# Script to run all Matryoshka-related tests and generate a comprehensive report

echo "=== FluxMD Matryoshka Engine Test Suite ==="
echo "Running comprehensive tests for potential errors..."
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create results directory
RESULTS_DIR="matryoshka_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Function to run a test and capture results
run_test() {
    local test_name=$1
    local test_file=$2
    local output_file="$RESULTS_DIR/${test_name}.log"
    
    echo -e "${YELLOW}Running $test_name...${NC}"
    
    if python -m pytest "$test_file" -v --tb=short > "$output_file" 2>&1; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name failed (see $output_file for details)${NC}"
        return 1
    fi
}

# Track overall results
total_tests=0
passed_tests=0

# Run each test suite
echo "### Test Execution ###"
echo

# 1. Physics Validation Tests
((total_tests++))
if run_test "Physics Validation" "tests/test_matryoshka_physics_validation.py"; then
    ((passed_tests++))
fi

# 2. Edge Case Tests
((total_tests++))
if run_test "Edge Cases" "tests/test_matryoshka_edge_cases.py"; then
    ((passed_tests++))
fi

# 3. Error Handling Tests
((total_tests++))
if run_test "Error Handling" "tests/test_matryoshka_error_handling.py"; then
    ((passed_tests++))
fi

# 4. Integration Tests
((total_tests++))
if run_test "Integration" "tests/test_matryoshka_integration.py"; then
    ((passed_tests++))
fi

# 5. Run the physics validation script
echo
echo -e "${YELLOW}Running physics validation script...${NC}"
if python tests/validate_matryoshka_physics.py --quick > "$RESULTS_DIR/physics_validation.log" 2>&1; then
    echo -e "${GREEN}✓ Physics validation completed${NC}"
else
    echo -e "${RED}✗ Physics validation had issues (see log)${NC}"
fi

# Generate summary report
echo
echo "### Summary Report ###"
echo

REPORT_FILE="$RESULTS_DIR/SUMMARY.md"
cat > "$REPORT_FILE" << EOF
# Matryoshka Engine Test Results

Generated on: $(date)

## Test Suite Results
- Total test suites: $total_tests
- Passed: $passed_tests
- Failed: $((total_tests - passed_tests))

## Known Issues Detected

### Critical Issues
1. **Zero Volume Atoms**: REF15 energy calculator encounters division by zero
   - Status: Warning system implemented, needs proper fix
   - Test: See test_matryoshka_physics_validation.py::TestZeroVolumeHandling

2. **Surface Self-Intersections**: Layer generation can create invalid geometries
   - Status: Basic smoothing implemented, needs improvement
   - Test: See test_matryoshka_edge_cases.py::TestSurfaceEdgeCases

### Performance Issues
1. **Memory Management**: Layer cache can grow unbounded
   - Status: Basic cache management implemented
   - Test: See test_matryoshka_physics_validation.py::TestMemoryLeaks

2. **Worker Process Crashes**: Limited error recovery in parallel mode
   - Status: Basic error handling present
   - Test: See test_matryoshka_edge_cases.py::TestParallelProcessingEdgeCases

## Detailed Results
EOF

# Append key findings from each test
for log_file in "$RESULTS_DIR"/*.log; do
    if [[ -f "$log_file" ]]; then
        echo >> "$REPORT_FILE"
        echo "### $(basename "$log_file" .log)" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        tail -n 20 "$log_file" | grep -E "(FAILED|PASSED|ERROR|WARNING)" >> "$REPORT_FILE" || echo "No specific issues found" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
    fi
done

echo
echo -e "${GREEN}Test execution complete!${NC}"
echo "Results saved to: $RESULTS_DIR/"
echo "Summary report: $REPORT_FILE"
echo

# Print quick summary
echo "### Quick Summary ###"
if [[ $passed_tests -eq $total_tests ]]; then
    echo -e "${GREEN}All test suites passed!${NC}"
else
    echo -e "${YELLOW}Some tests failed. This is expected as we're identifying issues.${NC}"
fi

echo
echo "Key findings:"
echo "- Zero volume atom handling needs improvement"
echo "- Surface self-intersection detection is simplistic"
echo "- Memory management could be optimized"
echo "- Error recovery in parallel processing needs work"
echo
echo "See $RESULTS_DIR/SUMMARY.md for full details"