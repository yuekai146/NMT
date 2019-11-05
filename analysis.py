import argparse


def remove_bpe(s):
    return (s + ' ').replace("@@ ", "").rstrip()


def remove_special_tok(s):
    s = s.replace("<s>", "")
    s = s.replace("</s>", "")
    s = s.replace("<pad>", "")
    s = s.replace("<unk>", "")
    return s


def read_sents(fpath, bpe=True):
    lines = open(fpath, 'r').read().split('\n')[:-1]
    if bpe:
        lines = [remove_special_tok(remove_bpe(l)).strip()  for l in lines]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None,
            help='Text file to analyze'
            )
    parser.add_argument('-o', '--output', type=str, default=None,
            help="Path to store analysis result"
            )
    args = parser.parse_args()
    sents = read_sents(args.input)
    # Sentence lengths
    lengths = [len(l.split(' ')) for l in sents]

    # Vocabulary size
    words = list(set(' '.join(sents).split(' ')))

    # Sentence diversity
    diversity = []
    for l in sents:
        diversity.append(len(set(l.split(' '))) / len(l.split(' ')))

    # Word distribution
    pass

    lengths_output = args.output + '_lengths.txt'
    vocab_size_output = args.output + '_vocab_size.txt'
    diversity_output = args.output + '_diversity.txt'

    f = open(lengths_output, 'w')
    f.write('\n'.join([str(item) for item in lengths]) + '\n')
    f.close()

    f = open(vocab_size_output, 'w')
    f.write(str(len(words)))
    f.close()
   
    f = open(diversity_output, 'w')
    f.write('\n'.join([str(item) for item in diversity]) + '\n')
    f.close()


if __name__ == "__main__":
    main()
