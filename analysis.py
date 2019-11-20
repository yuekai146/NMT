import argparse
import os


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
    parser.add_argument('--data_dir', type=str, default=None,
            help="Path of total training data"
            )
    parser.add_argument('-l', '--lan', type=str, default=None,
            help="What language is been analyzed now."
            )
    args = parser.parse_args()
    sents = read_sents(args.input)
    # Sentence lengths
    lengths = [len(l.split(' ')) for l in sents]
    avg_len = sum(lengths) / len(lengths)

    # Vocabulary size
    words = set(' '.join(sents).split(' ')[:-1])
    total = read_sents(os.path.join(args.data_dir, 'train.' + args.lan))
    total_words = set(' '.join(total).split(' ')[:-1])
    coverage = len(words.intersection(total_words)) / len(total_words)
    
    test = read_sents(os.path.join(args.data_dir, 'test.' + args.lan))
    test_words = set(' '.join(test).split(' ')[:-1])
    test_coverage = len(words.intersection(test_words)) / len(test_words)

    # Word distribution
    pass
    
    '''
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
    '''
    print("Average length:{}".format(avg_len))
    print("Word coverage:{}".format(coverage))
    print("Test coverage:{}".format(test_coverage))


if __name__ == "__main__":
    main()
