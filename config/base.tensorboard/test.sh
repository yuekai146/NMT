TEXT_DIR=/mnt/data/zhaoyuekai/active_NMT/playground/de-en/iwslt14_de_en
CUDA_VISIBLE_DEVICES=1 python translate.py -ckpt checkpoints/checkpoint_best_ppl.pth \
	-text $TEXT_DIR/test.de >> ref.en
