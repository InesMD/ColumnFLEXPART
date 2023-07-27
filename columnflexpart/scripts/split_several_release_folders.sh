
for folder in $(find $1original/ -maxdepth 1 -mindepth 1 -type d -printf '%f\n')
do 
	#mkdir $1splitted_test/$folder/
	python split_releases.py $1original/$folder/ $1splitted_test/$folder/ -r
done

#for dir in $1/*/ 
#do 
#	python split_releases.py dir $3/dir -r
#done
