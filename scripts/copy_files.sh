#! /bin/bash
cat $1| while read LINE; do
    echo $LINE
    cp ../../styleGAN2/dataset/CelebAMask-HQ/CelebA-HQ-img/$LINE ../dataset/celebA/images/
done