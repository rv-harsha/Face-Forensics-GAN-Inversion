#!/bin/bash 

files=()
while read p; do
    files+=($p)
done <$1

# Iterate the loop to read and print each array element
for value in "${files[@]}"
do
    echo $value
done