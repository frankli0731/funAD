#!/usr/bin/env bash
# File       : check_coverage.sh
# Description: Coverage wrapper around test suite driver script

set -e

tool='coverage'
if [[ $# -gt 0 ]]; then
    # optional argument to use different tool to check coverage
    tool="${1}"; shift
fi

if [[ ${tool} == 'coverage' ]]; then
    # run the tests (generates coverage data to build report)
    ./run_tests.sh coverage run --source=funAD "${@}"

    # build the coverage report on stdout
    coverage report -m
elif [[ ${tool} == 'pytest' ]]; then
    # generate coverage reports with pytest in one go
    ./run_tests.sh pytest --cov=funAD "${@}"
else
    # error: write to stderr
    >&2 echo "Error: unknown tool '${tool}'"
    exit 1
fi

