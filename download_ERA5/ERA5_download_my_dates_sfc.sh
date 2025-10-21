#!/bin/bash
# We will download every few days so they aren't too correlated
# over N years so that this turns out to be 1Tb?
dir="/zhome/98/d/202490/Documents/Thesis/ClimateDiffuse/data/"

year_start=1953
month_start=1

year_end=1957
month_end=12

for year in $(seq ${year_start} 1 ${year_end}); do
    for m in $(seq ${month_start} 1 ${month_end}); do
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
        echo "Getting surface files for $year $month"
        source ERA5_download_project.sh ${year} ${month} ${last_day}
        echo "Subsample in time"
        python preprocessing_subsample.py --year ${year} --month ${month} --last_day ${last_day} --remove_files  
        echo "Done for ${year} ${month}"
    done
done
echo DONE

