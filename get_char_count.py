import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-tv', '--total_vocab', type=str,
            help="Unfiltered vocab with rare words"
            )
    parser.add_argument(
            '-lf', '--least_frequency', type=int,
            help="Keep words that appear more than this freqency"
            )
    parser.add_argument(
            '-os', '--orig_src', type=str,
            help="Unfiltered source training corpus"
            )
    parser.add_argument(
            '-ot', '--orig_tgt', type=str,
            help="Unfiltered target training corpus"
            )
    args = parser.parse_args()
    
    f = open(args.total_vocab, 'r')
    words = f.read().split('\n')[:-1]
    f.close()

    # Get chars
    words_ = [w.split(' ')[0] for w in words]
    words_ = ''.join(words_)
    chars = list(set(words_))

    # Get count
    count = {}
    for l in words:
        w, n = l.split(' ')
        n = int(n)
        c_in_w = list(set(w))

        for c in c_in_w:
            if c in count:
                count[c] += (len(w) - len(w.replace(c, ''))) * n
            else:
                count[c] = (len(w) - len(w.replace(c, ''))) * n
    count = sorted(count.items(), key=lambda item: -item[1])
    chars = [c[0] for c in count if c[1] > args.least_frequency]
    print(chars)

    f_src = open(args.orig_src, 'r')
    f_tgt = open(args.orig_tgt, 'r')
    l_src = f_src.readline()
    l_tgt = f_tgt.readline()
    while len(l_src) != 0:
        l_chars = set(l_src.rstrip().replace(' ', ''))
        tgt_l_chars = set(l_tgt.rstrip().replace(' ', ''))
        if l_chars.issubset(chars) and tgt_l_chars.issubset(chars):
            print("S: ", l_src.rstrip())
            print("T: ", l_tgt.rstrip())
        l_src = f_src.readline()
        l_tgt = f_tgt.readline()

    f_src.close()
    f_tgt.close()

if __name__ == "__main__":
    main()
