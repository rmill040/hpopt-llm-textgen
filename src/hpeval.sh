#!/usr/bin/env bash

ls *.json | xargs -I {} python hpeval.py --path {} --sample-size 1.0 