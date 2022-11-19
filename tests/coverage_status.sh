#!/usr/bin/env bash
# File       : check_coverage.sh
#!/usr/bin/env bash
# File       : check_coverage.sh
# Description: Coverage wrapper around test suite driver script
# Copyright 2022 Harvard University. All Rights Reserved.

BAR=30

./run_tests.sh pytest --cov-fail-under=90 --cov-report term-missing --cov=funAD > result.log

if [[ $(awk '$1 == "TOTAL" {print $NF+0}' result.log) -ge $BAR ]]; then
    exit 0
else
    exit 1
fi