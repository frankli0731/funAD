#!/usr/bin/env bash
tests = (
    test_function.py
)
export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}
if [[ $# -gt 0 && ${1} == 'coverage']]; then
	driver = "${@} -m unittest"
elif [[ $# -gt 0 && ${1} == 'pytest' ]]; then
	driver = "${@}"
elif [[ $# -gt 0 && ${1} == 'CI' ]]; then
	shift
	unset PYTHONPATH
	driver = "pytest ${@}"
else
	driver = "python ${@} -m unittest"
fi

${driver} ${tests[@]}