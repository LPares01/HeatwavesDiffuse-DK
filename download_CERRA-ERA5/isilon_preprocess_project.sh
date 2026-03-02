#!/bin/bash

grid_file=/dmidata/users/lucpar/MASTER_THESIS/projects/ClimateDiffuse_sync/download_CERRA-ERA5/cerra_grid.txt
src_path=/net/isilon/ifs/arch/home/jis
datasets=(cerra era5)
stats=(max)

# Working configuration
dest_path=/dmidata/users/lucpar/MASTER_THESIS/data
years=({1985..2020})
months=({01..12})

# Testing configuration
# dest_path=/dmidata/users/lucpar/MASTER_THESIS/data/test
# years=1985
# months=04
# echo "Testing"

for year in ${years[@]}; do

	for month in ${months[@]}; do

		for ds in ${datasets[@]}; do

			echo "${year} - ${month} - ${ds^^}"

			cur_dir=${ds^^}/${year}
			mkdir -p ${dest_path}/${cur_dir}
			src_file=${src_path}/${cur_dir}/${ds}_hourly_${year}${month}.grb
			t2m_file=${dest_path}/${cur_dir}/${ds}_t2m_${year}${month}.nc4c

			if [ ${ds} == cerra ]; then
				cdo -f nc -selindexbox,433,688,534,789 -selvar,2t ${src_file} ${t2m_file}
			else
				cdo -f nc -sellonlatbox,-6,26,48,64 -selvar,2t ${src_file} ${t2m_file}
			fi

			for stat in ${stats[@]}; do

				echo -e "Computing T${stat}."
				stat_path=${dest_path}/${cur_dir}/${stat}/
				mkdir -p ${stat_path}
				dest_file=${stat_path}/${ds}_t2m_${stat}_${year}${month}.nc4c
				if [ ${ds} == cerra ]; then
					cdo -day${stat} -selmonth,${month} ${t2m_file} ${dest_file}
				else
					cdo -remapbil,${grid_file} -day${stat} -selmonth,${month} ${t2m_file} ${dest_file}
				fi
			done
		done
	done
done