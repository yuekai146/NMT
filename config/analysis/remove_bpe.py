import argparse

def remove_bpe(s):
    return (s + ' ').replace("@@ ", "").rstrip()

def remove_special_tok(s):
    s = s.replace("<s>", "")
    s = s.replace("</s>", "")
    s = s.replace("<pad>", "")
    s = s.replace("<unk>", "")
    return s

parser = argparse.ArgumentParser()
parser.add_argument(
        "-i", "--input_path", type=str, 
        help="where to read data", required=True
        )
parser.add_argument(
        "-o", "--output_path", type=str, 
        help="where to write result", required=True
        )
args = parser.parse_args()

f = open(args.input_path, 'r')
text = f.read().split('\n')
if text[-1] == "":
    text = text[:-1]
f.close()


punc = [".", ",", "?", "!", "'", "<", ">", ":", ";", "(", ")", 
        "{", "}", "[", "]", "-", "..", "...", "...."]

count = {}
text_without_bpe = []
for s in text:
    text_without_bpe.append(remove_special_tok(remove_bpe(s)))
#for s in text_without_bpe:
#    sentence = s.split()
#    for token in sentence:
#        if token not in punc:
#            if token in count.keys():
#                count[token] += 1
#            else:
#                count[token] = 1

f = open(args.output_path, 'w')
for s in text_without_bpe:
    f.write(s + '\n')
f.close()

