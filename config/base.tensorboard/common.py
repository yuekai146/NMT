import torch

class Config:
    encoder_num_layers = 6
    decoder_num_layers = 6
    d_model = 512
    d_ff = 1024
    num_heads = 4
    dropout = 0.3
    label_smoothing = 0.1
    
    BOS = "<s>"
    EOS = "</s>"
    PAD = "<pad>"
    UNK = "<unk>"
    SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
    N_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
    MAX_LEN = 250 
    
    # data paths
    DATA_PATH="/mnt/data/zhaoyuekai/active_NMT/playground/de-en"
    SRC_RAW_TRAIN_PATH = DATA_PATH + "/iwslt14_de_en/train.de"
    TGT_RAW_TRAIN_PATH = DATA_PATH + "/iwslt14_de_en/train.en"
    SRC_RAW_VALID_PATH = DATA_PATH + "/iwslt14_de_en/valid.de"
    TGT_RAW_VALID_PATH = DATA_PATH + "/iwslt14_de_en/valid.en"
    SRC_VOCAB_PATH = DATA_PATH + "/iwslt14_de_en/vocab.de"
    TGT_VOCAB_PATH = DATA_PATH + "/iwslt14_de_en/vocab.en"
    src_n_vocab = len(open(SRC_VOCAB_PATH, 'r').read().split('\n')) + N_SPECIAL_TOKENS
    tgt_n_vocab = len(open(TGT_VOCAB_PATH, 'r').read().split('\n')) + N_SPECIAL_TOKENS
    SRC_LAN = "de"
    TGT_LAN = "en"
    VALID_RATIO = 0.001
    NGPUS = 1
    BATCH_SIZE = 128 * NGPUS
    tokens_per_batch = 4096 * NGPUS # if tokens_per_batch > 0, ignore BATCH_SIZE
    max_batch_size = 1000 * NGPUS

    # For optimizer
    opt_factor = 2
    opt_warmup = 4000
    opt_init_lr = 0.0
    beta1 = 0.9
    beta2 = 0.98
    weight_decay = 0.0001
    opt_eps = 1e-9

    # For trainer
    use_cuda = torch.cuda.is_available()
    multi_gpu = True
    epoch_size = 40
    continue_path = None
    dump_path = "checkpoints/"
    tensorboard_log_path = "tb_log/"
    summary_interval = 5
    log_param_and_grad = False
    log_lr = True
    log_histograms = False
    clip_grad_norm = 0.0
    accumulate_gradients = 1
    save_periodic = 1
    valid_metrics = {"ppl":-1}
    init_metric = -1e12
    print_interval = 5


config = Config()
