#!/bin/bash
CODECARGS="--draco --no_tiling_first --no_tiling_rest"
CODECARGS=$CODECARGS" -onum_clusters_scaling=1024 -on_bits_proposal_scaling=n_bits_proposal_balanced_values(8)"
CODECARGS=$CODECARGS" -onum_clusters_rotation_re=256 -on_bits_proposal_rotation_re=n_bits_proposal_balanced_values(6)"
CODECARGS=$CODECARGS" -onum_clusters_rotation_im=1024 -on_bits_proposal_rotation_im=n_bits_proposal_balanced_values(8)"
CODECARGS=$CODECARGS" -onum_clusters_opacity=256 -on_bits_proposal_opacity=n_bits_proposal_balanced_values(6)"
CODECARGS=$CODECARGS" -onum_clusters_features_dc=512 -on_bits_proposal_features_dc=n_bits_proposal_balanced_values(7)"
CODECARGS=$CODECARGS" -onum_clusters_features_rest=[256,128,64]"
CODECARGS=$CODECARGS" -on_bits_proposal_features_rest=[n_bits_proposal_balanced_values(6),n_bits_proposal_balanced_values(5),n_bits_proposal_balanced_values(4)]"
encode() {
    python -m cags.encode \
        -s output/$1/$2 \
        -d output/$1-compressed/$2 \
        --source_init output/$1/frame1 \
        --destination_init output/$1-compressed/frame1 \
        --iteration_init $3 -i $4 \
        --frame_start $5 --frame_end $6 \
        $CODECARGS
}
# encode coffee_martini refine/masked-propagate-dot-cotracker3 10000 1000 2 300 #debug

METHOD=refine/masked-propagate-dot-cotracker3
encode walking $METHOD 10000 1000 2 75
encode taekwondo $METHOD 10000 1000 2 101
encode boxing $METHOD 10000 1000 2 71

encode coffee_martini $METHOD 10000 1000 2 300
encode cook_spinach $METHOD 10000 1000 2 300
encode cut_roasted_beef $METHOD 10000 1000 2 300
encode flame_salmon_1 $METHOD 10000 1000 2 1200
encode flame_steak $METHOD 10000 1000 2 300
encode sear_steak $METHOD 10000 1000 2 300

encode discussion $METHOD 10000 1000 2 300
encode stepin $METHOD 10000 1000 2 300
encode trimming $METHOD 10000 1000 2 300
encode vrheadset $METHOD 10000 1000 2 300

encode basketball $METHOD 10000 1000 2 150
encode boxes $METHOD 10000 1000 2 150
encode football $METHOD 10000 1000 2 150
encode juggle $METHOD 10000 1000 2 150
encode softball $METHOD 10000 1000 2 150
encode tennis $METHOD 10000 1000 2 150
