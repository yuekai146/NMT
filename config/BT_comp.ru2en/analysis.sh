function run () {
	method=$1
	i=$2
	DATA_DIR=/data/NMT/data/de-en/wmt17_de_en
	
	mkdir -p data_analysis/$method/$i/de-en data_tmp
	aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp s3://zhaoyuekai/active_NMT/config/base.back_translation/result/$method/active_data/de-en/train_$i.de data_tmp/
	python3 analysis.py -i data_tmp/train_$i.de --data_dir $DATA_DIR -l de > data_analysis/$method/$i/de-en/de.summary

	rm data_tmp/train_$i.de

	aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp s3://zhaoyuekai/active_NMT/config/base.back_translation/result/$method/active_data/de-en/train_$i.en data_tmp/
	python3 analysis.py -i data_tmp/train_$i.en --data_dir $DATA_DIR -l en > data_analysis/$method/$i/de-en/en.summary

	rm data_tmp/train_$i.en
	
	mkdir -p data_analysis/$method/$i/en-de data_tmp
	aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp s3://zhaoyuekai/active_NMT/config/base.back_translation/result/$method/active_data/en-de/train_$i.de data_tmp/
	python3 analysis.py -i data_tmp/train_$i.de --data_dir $DATA_DIR -l de > data_analysis/$method/$i/en-de/de.summary

	rm data_tmp/train_$i.de

	aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp s3://zhaoyuekai/active_NMT/config/base.back_translation/result/$method/active_data/en-de/train_$i.en data_tmp/
	python3 analysis.py -i data_tmp/train_$i.en --data_dir $DATA_DIR -l en > data_analysis/$method/$i/en-de/en.summary

	rm data_tmp/train_$i.en
}


function main () {
	for method in random lc margin te tte; do
		for i in $( seq 1 11 ); do
			run $method $i
		done
	done	
}

main
