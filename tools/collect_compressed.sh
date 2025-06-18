#!/bin/bash
METHOD=refine/masked-propagate-dot-cotracker3
collect_all() {
    mkdir -p data/saved_compress/output/$1

    echo "Collecting output/$1-compressed/frame1"
    rm -rf data/saved_compress/output/$1/frame1
    cp -r output/$1-compressed/frame1 data/saved_compress/output/$1/frame1

    rm -rf data/saved_compress/output/$1/$METHOD
    mkdir -p data/saved_compress/output/$1/$METHOD
    for i in $(seq $2 $3); do
        echo "Collecting output/$1-compressed/$METHOD/frame$i"
        cp -r output/$1-compressed/$METHOD/frame$i data/saved_compress/output/$1/$METHOD/frame$i
    done
}
collect_all walking 2 75
collect_all taekwondo 2 101
collect_all boxing 2 71

collect_all coffee_martini 2 300
collect_all cook_spinach 2 300
collect_all cut_roasted_beef 2 300
collect_all flame_salmon_1 2 1200
collect_all flame_steak 2 300
collect_all sear_steak 2 300

collect_all discussion 2 300
collect_all stepin 2 300
collect_all trimming 2 300
collect_all vrheadset 2 300

collect_all basketball 2 150
collect_all boxes 2 150
collect_all football 2 150
collect_all juggle 2 150
collect_all softball 2 150
collect_all tennis 2 150

cd data/saved_compress
rm saved_compress.zip
zip -r saved_compress.zip ./output
