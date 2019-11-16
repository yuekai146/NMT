import torch


MODE = "MASS"


class MT_Config:

    encoder_num_layers = 6
    decoder_num_layers = 6
    d_model = 1024
    d_ff = 4096
    num_heads = 16
    dropout = 0.1
    label_smoothing = 0.1
    share_decoder_generator_embed = True
    share_all_embeddings = True
    gelu_activation = True
    
    BOS = "<s>"
    EOS = "</s>"
    PAD = "<pad>"
    UNK = "<unk>"
    MASK = "<mask>"
    SRC_LAN = "en"
    TGT_LAN = "de"

    SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, MASK, "<" + SRC_LAN.upper() + ">", "<" + TGT_LAN.upper() + ">"]
    N_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
    MAX_LEN = 250 
    
    # data paths
    DATA_PATH="../../data/de-en"
    SRC_RAW_TRAIN_PATH = DATA_PATH + "/wmt17_de_en/train.en"
    TGT_RAW_TRAIN_PATH = DATA_PATH + "/wmt17_de_en/train.de"
    SRC_RAW_VALID_PATH = DATA_PATH + "/wmt17_de_en/valid.en"
    TGT_RAW_VALID_PATH = DATA_PATH + "/wmt17_de_en/valid.de"
    if share_all_embeddings:
        SRC_VOCAB_PATH = DATA_PATH + "/wmt17_de_en/vocab.total"
        TGT_VOCAB_PATH = DATA_PATH + "/wmt17_de_en/vocab.total"
    else:
        SRC_VOCAB_PATH = DATA_PATH + "/wmt17_de_en/vocab.en"
        TGT_VOCAB_PATH = DATA_PATH + "/wmt17_de_en/vocab.de"

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
    SRC_LAN = "en"
    TGT_LAN = "de"
    BATCH_SIZE = 128
    tokens_per_batch = 3000 # if tokens_per_batch > 0, ignore BATCH_SIZE
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

    if fp16:
        if src_n_vocab % 8 != 0:
            src_n_vocab = (src_n_vocab // 8) * 8 + 8
        if tgt_n_vocab % 8 != 0:
            tgt_n_vocab = (tgt_n_vocab // 8) * 8 + 8

    # For trainer
    use_cuda = torch.cuda.is_available()
    multi_gpu = True
    epoch_size = 150
    continue_path = None
    dump_path = "checkpoints/"
    reload_network_only = True
    clip_grad_norm = 0.0
    accumulate_gradients = 2
    save_periodic = 1
    valid_metrics = {"ppl":-1}
    init_metric = -1e12
    print_interval = 5
    # To early stop if validation performance did not
    # improve for decrease_counts_max epochs
    decrease_counts_max = None
    stopping_criterion = "ppl"


class MASS_Config:
    encoder_num_layers = 6
    decoder_num_layers = 6
    d_model = 512
    d_ff = 2048
    num_heads = 8
    dropout = 0.1
    label_smoothing = 0.0
    share_decoder_generator_embed = True
    share_all_embeddings = True
    gelu_activation = True

    LANS = ['EN', 'DE']
    LANG2IDS = {}
    for i, k in enumerate(LANS):
        LANG2IDS[k] = i 
    BOS = "<s>"
    EOS = "</s>"
    PAD = "<pad>"
    UNK = "<unk>"
    MASK = "<mask>"
    LAN_IDS = ['<' + lan + '>' for lan in LANS]

    SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, MASK] + LAN_IDS
    N_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
    MAX_LEN = 250 
    
    # data paths
    DATA_PATH="../../data/de-en/news_crawl"
    MONO_RAW_TRAIN_PATH = []
    for lan in LANS:
        MONO_RAW_TRAIN_PATH.append(DATA_PATH + '/mono.' + lan.lower())

    valid_directions = 'de-en,en-de'
    VALID_DATA_PATH = '../../data/de-en/news_crawl'
    RAW_VALID_PATH = {}
    for direction in valid_directions.split(','):
        src, tgt = direction.split('-')
        assert src.upper() in LANS
        assert tgt.upper() in LANS
        RAW_VALID_PATH[direction] = (
                VALID_DATA_PATH + '/valid.' + direction + '.' + src, 
                VALID_DATA_PATH + '/valid.' + direction + '.' + tgt
                )
    assert share_all_embeddings
    TOTAL_VOCAB_PATH = DATA_PATH + "/vocab.total"

    data_bin = "data_bin/"
    train_iter_dump_path = data_bin + "train_iter"
    valid_iter_dump_path = data_bin + "valid_iter"
    total_vocab_dump_path = data_bin + "TOTAL"

    # Wrong n_vocab, should minus 1
    total_n_vocab = len(open(TOTAL_VOCAB_PATH, 'r').read().split('\n')[:-1]) + N_SPECIAL_TOKENS
    
    BATCH_SIZE = 128
    tokens_per_batch = 3000 * len(LANS) # if tokens_per_batch > 0, ignore BATCH_SIZE
    max_batch_size = 0

    # Masked language model
    word_mass = 0.5
    span_len = 100000
    mask_probs = [0.8, 0.1, 0.1]

    # For optimizer
    opt_warmup = 4000
    lr = 1e-4
    init_lr = None
    beta1 = 0.9
    beta2 = 0.98
    weight_decay = 0.0
    opt_eps = 1e-9

    # For fp16 training
    fp16 = True # Whether to use fp16 training
    amp = 2 # Level of optimization

    if fp16:
        if total_n_vocab % 8 != 0:
            total_n_vocab = (total_n_vocab // 8) * 8 + 8
    # For trainer
    use_cuda = torch.cuda.is_available()
    multi_gpu = True
    epoch_size = 50
    continue_path = None
    dump_path = "checkpoints/"
    reload_network_only = False
    optimizer_only = True
    clip_grad_norm = 0.0
    accumulate_gradients = 1
    save_periodic = 1
    valid_metrics = {}
    for direction in valid_directions.split(','):
        valid_metrics[direction.replace('-', '_') + '_ppl'] = -1
    init_metric = -1e12
    print_interval = 5
    # To early stop if validation performance did not
    # improve for decrease_counts_max epochs
    decrease_counts_max = None
    stopping_criterion = None


all_config = {
        "MT":MT_Config(),
        "MASS":MASS_Config()
        }
config = all_config[MODE]
