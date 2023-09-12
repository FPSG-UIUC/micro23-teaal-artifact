#!/bin/bash

runuser -l workspace -c 'cd scripts; accelergy -v1 -o tmp/init ../yamls/accelergy/simple.yaml > /dev/null'
runuser -l workspace -c 'cd scripts; python3 check.py'
