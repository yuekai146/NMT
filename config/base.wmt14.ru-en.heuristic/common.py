import torch

class Config:
    encoder_num_layers = 6
    decoder_num_layers = 6
    d_model = 512
    d_ff = 2048
    num_heads = 8
    dropout = 0.1
    label_smoothing = 0.1
    share_decoder_generator_embed = True
    share_all_embeddings = False
    gelu_activation = True
    
    BOS = "<s>"
    EOS = "</s>"
    PAD = "<pad>"
    UNK = "<unk>"
    SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
    N_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
    MAX_LEN = 250 
    SRC_LAN = "ru"
    TGT_LAN = "en"
    
    # data paths
    DATA_PATH="/data/ru-en/"
    SRC_RAW_TRAIN_PATH = DATA_PATH + "train." + SRC_LAN
    TGT_RAW_TRAIN_PATH = DATA_PATH + "train." + TGT_LAN
    SRC_RAW_VALID_PATH = DATA_PATH + "valid." + SRC_LAN
    TGT_RAW_VALID_PATH = DATA_PATH + "valid." + TGT_LAN
    if share_all_embeddings:
        SRC_VOCAB_PATH = DATA_PATH + "vocab.total"
        TGT_VOCAB_PATH = DATA_PATH + "vocab.total"
    else:
        SRC_VOCAB_PATH = DATA_PATH + "vocab." + SRC_LAN
        TGT_VOCAB_PATH = DATA_PATH + "vocab." + TGT_LAN

    data_bin = "data_bin/"
    train_iter_dump_path = data_bin + "train_iter"
    valid_iter_dump_path = data_bin + "valid_iter"
    src_vocab_dump_path = data_bin + "SRC"
    tgt_vocab_dump_path = data_bin + "TGT"

    # Wrong n_vocab, should minus 1
    src_n_vocab = len(open(SRC_VOCAB_PATH, 'r').read().split('\n')) - 1 + N_SPECIAL_TOKENS
    tgt_n_vocab = len(open(TGT_VOCAB_PATH, 'r').read().split('\n')) - 1 + N_SPECIAL_TOKENS
    
    if share_all_embeddings:
        assert src_n_vocab == tgt_n_vocab
    BATCH_SIZE = 128
    tokens_per_batch = 2048 # if tokens_per_batch > 0, ignore BATCH_SIZE
    max_batch_size = 0 

    # For optimizer
    opt_warmup = 4000
    lr = 7e-4
    init_lr = 1e-7
    beta1 = 0.9
    beta2 = 0.98
    weight_decay = 0.0
    opt_eps = 1e-9

    # For fp16 training
    fp16 = True # Whether to use fp16 training
    amp = 2 # Level of optimization

    # For trainer
    use_cuda = torch.cuda.is_available()
    multi_gpu = True
    epoch_size = 150
    continue_path = None
    dump_path = "checkpoints/"
    reload_network_only = False
    optimizer_only = True
    clip_grad_norm = 0.0
    accumulate_gradients = 2
    save_periodic = 1
    valid_metrics = {"ppl":-1}
    init_metric = -1e12
    print_interval = 5
    # To early stop if validation performance did not
    # improve for decrease_counts_max epochs
    decrease_counts_max = 10
    stopping_criterion = "ppl"

config = Config()
