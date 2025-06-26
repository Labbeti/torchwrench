#!/usr/bin/bash
# -*- coding: utf-8 -*-

docs_dpath=`dirname $0`
cd "$docs_dpath"

rm torchwrench.*rst
uv run sphinx-apidoc -e -M -o . ../src/torchwrench && uv run make clean && uv run make html

exit 0
