#!/bin/bash

for png in *.png; do
    convert "$png""[600x>]" "$png"
done

