#!/bin/bash

grids_path=/dmidata/users/lucpar/MASTER_THESIS/projects/ClimateDiffuse_sync/download_CERRA-ERA5
src_path=/net/isilon/ifs/arch/home/jis
dest_path=/dmidata/users/lucpar/MASTER_THESIS/data

datasets=(cerra era5)
# domains=(4.38,17.02,50.18,62.82 2.68,18.72,48.48,64.52)
years=({1985..2020})
months=({01..12})
stats=(mean max min)

# datasets=(cerra)
# # domains=(4.2,17.0,50.2,63.0)
# years=1985
# months=01
# stats=(mean max min)

i=0

for ds in ${datasets[@]}; do

	# domain=${domains[i]}
	grid=${grids_path}/${ds}_grid.txt

	for year in ${years[@]}; do

		cur_dir=${ds^^}/${year}
		mkdir -p ${dest_path}/${cur_dir}

		for month in ${months[@]}; do

			echo "${ds^^} - ${year} - ${month}"
			src_file=${src_path}/${cur_dir}/${ds}_hourly_${year}${month}.grb
			# nc_file=${dest_path}/${cur_dir}/${ds}_${year}${month}.nc4c
			t2m_file=${dest_path}/${cur_dir}/${ds}_t2m_${year}${month}.nc4c
			cdo -f nc -remapbil,${grid} -selvar,2t ${src_file} ${t2m_file}
			# cdo -sellonlatbox,${domain} ${nc_file} ${t2m_file}
			# rm ${nc_file}

			for stat in ${stats[@]}; do

				echo -e "\t${stat}"
				stat_dir=${cur_dir}/${stat}/
				mkdir -p ${dest_path}/${stat_dir}
				dest_file=${dest_path}/${stat_dir}/${ds}_t2m_${stat}_${year}${month}.nc4c
				cdo -day${stat} ${t2m_file} ${dest_file}

			done

		done

	done

	((i++))

done
