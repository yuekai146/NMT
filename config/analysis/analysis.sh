for method in random combine; do
	I=./data/$method/labeled_12.ru
	cd result
	mkdir $method
	cd ..
	O=./result/$method/remove_bpe.txt
	voc=./result/$method/vocab.txt
	res=./result/$method/result_dis.txt

	python3 remove_bpe.py -i $I -o $O

	/data/fastBPE/fast getvocab $O > $voc

	python3 get_dis.py -i $voc -o $res
	cd result/$method
	rm remove_bpe.txt
	rm vocab.txt
	cd ..
	cd ..
done	
