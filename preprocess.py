from collections import Counter
import pickle as pkl
import re


#----------------------------
# Create dictionary
# text: [sentence] * N
#----------------------------
def create_dict(text, max_vocab_dim=16384):
	data = []
	for t in text:
		data += t.split()

	counter = Counter(data)
	w2i = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "<sos>":3}

	for i, (w, c) in enumerate(counter.most_common(max_vocab_dim-4), 4):
		w2i[w] = i

	i2w = {i: w for w, i in w2i.items()}

	return w2i, i2w


#----------------------------
# Create skip-gram
# text: [sentence] * N
#----------------------------
def create_skipgram(text, w2i, window_size=2):
	x = []
	y = []

	for s in text:
		length = len(s)

		for i in range(length):
			if s[i] == w2i["<unk>"]: continue

			for j in range(max(i-window_size, 0), min(i+window_size+1, length)):
				if i != j and s[j] != w2i["<unk>"]:
					x.append(s[i])
					y.append(s[j])

	return x, y


#----------------------------
# Normalize text
# text: [sentence] * N
#----------------------------
def normalize_text(text, regex=r"([\?\!,.\/\"\(\)\'`;:]|\.+)"):
	text = [t.strip().lower() for t in text]
	text = [re.sub(regex, r" \1 ", t) for t in text]

	return text


#Load data
src_path = "./dataset/kit.txt"

with open(src_path, "r", encoding="utf-8") as fp:
	src_text = fp.read().split("\n")


#Process text
src_text = normalize_text(src_text)
src_v2i, src_i2v = create_dict(src_text)
src_token = [[src_v2i.get(w, src_v2i["<unk>"]) for w in s.split()] for s in src_text]

for sent in src_token:
	if len(sent) > 0 and sent[-1] != src_v2i["."]:
		sent.append(src_v2i["."])

skipgram_x, skipgram_y = create_skipgram(src_token, src_v2i)

print("vocab_dim = {:d}".format(len(src_v2i)))
print("# sentences = {:d}".format(len(src_token)))
print("# skipgram = {:d}".format(len(skipgram_x)))

#Save dataset
pkl.dump((src_token, src_v2i, src_i2v), open("./dataset/src_dataset.pkl", "wb"))
pkl.dump((skipgram_x, skipgram_y), open("./dataset/src_skipgram.pkl", "wb"))