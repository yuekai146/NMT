import math
import argparse

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
text = f.read().split('\n')[:-1]
f.close()

punc = [".", ",", "?", "!", "'", "<", ">", ":", ";", "(", ")", 
        "{", "}", "[", "]", "-", "..", "...", "...."]

dis = []
total_count = 0

select_word = 300000

for i in range(min(select_word, len(text))):
    s = text[i]
    if len(s) > 0:
        word = s.split()[0]
        count = math.log(float(s.split()[1]) + 1)
        if word not in punc:
            total_count += count

for i in range(min(select_word, len(text))):
    s = text[i]
    if len(s) > 0:
        word = s.split()[0]
        count = math.log(float(s.split()[1]) + 1)
        if word not in punc:
            percent = count/total_count
            dis.append((word, percent))

res = []
total_num = 10

for i in range(len(dis), select_word):
    dis.append((' ', 0))

num_per_label = select_word // total_num
for i in range(total_num):
    label_num = 0
    for j in range(num_per_label):
        label_num += dis[i * num_per_label + j][1]
    res.append(label_num)

f = open(args.output_path, 'w')
for per in res:
    f.write(str(per) + '\n')
f.close()


