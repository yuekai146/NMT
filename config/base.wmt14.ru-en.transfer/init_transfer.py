import random


def read_sents(fpath):
    lines = open(fpath, 'r').read().split('\n')[:-1]
    return lines


def write_sents(lines, fpath):
    f = open(fpath, 'w')
    f.write('\n'.join(lines) + '\n')
    f.close()


def main():
    ru = read_sents('/data/rufr-en/ru-en/train.ru')
    en = read_sents('/data/rufr-en/ru-en/train.en')


    combined = list(zip(ru, en))
    random.shuffle(combined)

    ru[:], en[:] = zip(*combined)


    labeled_1_ru, labeled_1_en = [], []
    unlabeled_1_ru, unlabeled_1_en = [], []
    n_tok = 0
    for s_ru, s_en in zip(ru, en):
        ru_toks = len((s_ru + ' ').replace("@@ ", "").rstrip().split(' '))
        if ru_toks + n_tok <= 1800000:
            labeled_1_ru.append(s_ru)
            labeled_1_en.append(s_en)
            n_tok += ru_toks
        else:
            unlabeled_1_ru.append(s_ru)
            unlabeled_1_en.append(s_en)

    write_sents(labeled_1_ru, 'init/labeled_1.ru')
    write_sents(labeled_1_en, 'init/labeled_1.en')
    write_sents(unlabeled_1_ru, 'init/unlabeled_1.ru')
    write_sents(unlabeled_1_en, 'init/unlabeled_1.en')


if __name__ == "__main__":
    main()
