import torch

class Config:
    encoder_num_layers = 6
    decoder_num_layers = 6
    d_model = 512
    d_ff = 1024
    num_heads = 4
    dropout = 0.1
    label_smoothing = 0.1
    share_decoder_generator_embed = True
    share_all_embeddings = False
    
    BOS = "<s>"
    EOS = "</s>"
    PAD = "<pad>"
    UNK = "<unk>"
    SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
    N_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
    MAX_LEN = 250 
    
    # data paths
    DATA_PATH="../../data/de-en"
    SRC_RAW_TRAIN_PATH = DATA_PATH + "/iwslt14_de_en/train.de"
    TGT_RAW_TRAIN_PATH = DATA_PATH + "/iwslt14_de_en/train.en"
    SRC_RAW_VALID_PATH = DATA_PATH + "/iwslt14_de_en/valid.de"
    TGT_RAW_VALID_PATH = DATA_PATH + "/iwslt14_de_en/valid.en"
    if share_all_embeddings:
        SRC_VOCAB_PATH = DATA_PATH + "/iwslt14_de_en/vocab.total"
        TGT_VOCAB_PATH = DATA_PATH + "/iwslt14_de_en/vocab.total"
    else:
        SRC_VOCAB_PATH = DATA_PATH + "/iwslt14_de_en/vocab.de"
        TGT_VOCAB_PATH = DATA_PATH + "/iwslt14_de_en/vocab.en"

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
    SRC_LAN = "de"
    TGT_LAN = "en"
    BATCH_SIZE = 128
    tokens_per_batch = 4096 # if tokens_per_batch > 0, ignore BATCH_SIZE
    max_batch_size = 1000 

    # For optimizer
    opt_warmup = 4000
    lr = 7e-4
    init_lr = 1e-7
    beta1 = 0.9
    beta2 = 0.98
    weight_decay = 0.0001
    opt_eps = 1e-9

    # For fp16 training
    fp16 = True # Whether to use fp16 training
    amp = 2 # Level of optimization

    if fp16:
        if src_n_vocab % 8 != 0:
            src_n_vocab = (src_n_vocab // 8) * 8 + 8
        if tgt_n_vocab % 8 != 0:
            tgt_n_vocab = (tgt_n_vocab // 8) * 8 + 8

    # For trainer
    use_cuda = torch.cuda.is_available()
    multi_gpu = True
    epoch_size = 15
    continue_path = None
    dump_path = "checkpoints/"
    reload_network_only = True
    clip_grad_norm = 0.0
    accumulate_gradients = 1
    save_periodic = 1
    valid_metrics = {"ppl":-1}
    init_metric = -1e12
    print_interval = 5
    # To early stop if validation performance did not
    # improve for decrease_counts_max epochs
    decrease_counts_max = 25
    stopping_criterion = "ppl"

config = Config()
