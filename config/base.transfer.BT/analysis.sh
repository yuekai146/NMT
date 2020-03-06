function run () {
	method=$1
	i=$2
	DATA_DIR=/data/NMT/data/de-en/wmt17_de_en
	
	mkdir -p data_analysis/$method/$i data_tmp
	aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp s3://zhaoyuekai/active_NMT/config/base.transfer/$method/active_data/labeled_$i.de data_tmp/
	python3 analysis.py -i data_tmp/labeled_$i.de --data_dir $DATA_DIR -l de > data_analysis/$method/$i/de.summary

	rm data_tmp/labeled_$i.de

	aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp s3://zhaoyuekai/active_NMT/config/base.transfer/$method/active_data/labeled_$i.en data_tmp/
	python3 analysis.py -i data_tmp/labeled_$i.en --data_dir $DATA_DIR -l en > data_analysis/$method/$i/en.summary

	rm data_tmp/labeled_$i.en
}


function main () {
	for method in random lc margin te tte longest shortest; do
		for i in $( seq 1 12 ); do
			run $method $i
		done
	done	
}

main
