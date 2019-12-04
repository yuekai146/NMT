function run () {
	method=$1
	i=$2
	DATA_DIR=/data/NMT/data/de-en/wmt17_de_en
	
	mkdir -p data_analysis/$method/$i/de-en
	python3 analysis.py -i result/te-dden/active_data/de-en/train_$i.de --data_dir $DATA_DIR -l de > data_analysis/$method/$i/de-en/de.summary
	python3 analysis.py -i result/te-dden/active_data/de-en/train_$i.en --data_dir $DATA_DIR -l en > data_analysis/$method/$i/de-en/en.summary

	mkdir -p data_analysis/$method/$i/en-de
	python3 analysis.py -i result/te-dden/active_data/en-de/train_$i.de --data_dir $DATA_DIR -l de > data_analysis/$method/$i/en-de/de.summary
	python3 analysis.py -i result/te-dden/active_data/en-de/train_$i.en --data_dir $DATA_DIR -l en > data_analysis/$method/$i/en-de/en.summary

}


function main () {
	for i in $( seq 1 11 ); do
		run combine $i
	done
}

main
