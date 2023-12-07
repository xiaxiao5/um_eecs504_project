#!/bin/bash
# This script renames the files in the source directory to the format 

# directory where the source files are located
src_dir="/tmp/pv_frames"

# directory where the files will be copied to
dest_dir="/tmp/pv_frames_renamed"

# iterate over all jpg files in the source directory
for src_file in "$src_dir"/frame_*.jpg; do
    # extract the frame number from the filename
    frame_num=$(basename "$src_file" .jpg | cut -d'_' -f 2)

    # decrement the frame number
    # ((frame_num--))
    frame_num=$((10#$frame_num - 1))

    # create the destination filename
    dest_file="$dest_dir/frame_$(printf "%010d" "$frame_num").jpg"

    # copy the file
    cp "$src_file" "$dest_file"
    
done

