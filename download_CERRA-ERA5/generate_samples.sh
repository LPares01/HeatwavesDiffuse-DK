#!/bin/bash
# We create random subsamples from individual months then concatenate by year

dir=../data/CERRA-ERA5/

# Full dataset
dummy=0
year_start=1985
year_end=2014

# Dummy dataset
# dummy=1
# year_start=1985
# year_end=1990

for year in $(seq ${year_start} 1 ${year_end}); do

    for m in {1..12}; do

        if [ "${m}" = 4 ] || [ "${m}" = 6 ] || [ "${m}" = 9 ] || [ "${m}" = 11 ] ; then
            last_day=30
        elif [ "${m}" = 2 ]; then

            if [ "$(((${year}) % 4))" = 0 ]; then
		       last_day=29
	        else last_day=28
	        fi

        else last_day=31
        fi

        month=$(printf "%02d" ${m})
        
        echo "Shuffling days"
        if [ ${dummy} == 0 ]; then
            python CERRA-ERA5_preprocessing_subsample.py --year ${year} --month ${m} --last_day ${last_day} --dir ${dir}
        else
            python CERRA-ERA5_preprocessing_subsample.py --year ${year} --month ${m} --last_day ${last_day} --dir ${dir} --dummy
        fi
        echo "Done for ${year} ${month}"
    done

    echo "Concatenating all months for ${year}"
    if [ ${dummy} == 0 ]; then
        python CERRA-ERA5_preprocessing_concat_year.py --year ${year} --dir ${dir} --remove_files
    else
        python CERRA-ERA5_preprocessing_concat_year.py --year ${year} --dir ${dir}dummy/ --remove_files
    fi
    echo "Done for ${year}"

done
echo DONE
