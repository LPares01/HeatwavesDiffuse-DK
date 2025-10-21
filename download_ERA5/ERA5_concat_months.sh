#!/bin/bash
# We concatenate samples from individual months by year

dir="/zhome/98/d/202490/Documents/Thesis/ClimateDiffuse/data/"

year_start=1954

year_end=1957

for year in $(seq ${year_start} 1 ${year_end}); do
    echo "Concatenate all months for ${year}"
    python preprocessing_concat_year.py --year ${year} --data ${dir} --remove_files
    echo "Done for ${year}"
done
echo DONE
