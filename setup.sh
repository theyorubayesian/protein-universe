kaggle datasets download -d googleai/pfam-seed-random-split
unzip pfam-seed-random-split.zip && rm -r random_split/random_split
mv random_split data && rm pfam*.zip
