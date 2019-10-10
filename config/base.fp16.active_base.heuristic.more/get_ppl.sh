N_ROUNDS=12

for i in $( seq 1 $N_ROUNDS );
do
	cat $i/train.log | grep best | tail -n 2 | head -n 1 | awk '{print $NF}'
done
