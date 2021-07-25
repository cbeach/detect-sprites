#!/usr/bin/env bash

file_name="profile_data.$(date --iso-8601).pyprof"
python -m cProfile -o $file_name  simple_patch.py && pyprof2calltree -i $file_name -k

