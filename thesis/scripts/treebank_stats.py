import sys
import math
from collections import Counter
import numpy as np

blokk = []
deprels_to_count = []

m_index = 0
# i-index
number_of_sentences = 0
current_lang = None
switches = 0
total_i_index = 0
# burstiness
previous_lang = None
current_sample = 0
samples = []
init_vector = []
final_vector = []

for line in sys.stdin:
    if line[0] == '#':
        continue

    if not line.rstrip():
        # m-index
        langs = set()
        tokens_per_lang = {}
        # ==
        number_of_sentences += 1
        current_lang = None
        current_sample = 0
        # do stuff here
        for row in blokk:
            # m-index
            lang = row[9]
            langs.add(lang)
            try:
                tokens_per_lang[lang] += 1
            except:
                tokens_per_lang[lang] = 1
            # ---

            # i-index
            lang = row[9]
            if not current_lang:
                current_lang = lang
            else:
                if lang != current_lang:
                    switches += 1
            # ---

            # burstiness
            lang = row[9]
            if lang != current_lang:
                samples.append(current_sample)
                current_sample = 1

            else:
                current_sample += 1 
            # ---
            current_lang = lang

            head = int(row[6]) - 1
            lang = row[9]
            if head != -1:
                parent_lang = blokk[head][9]
                if parent_lang != lang:
                    deprels_to_count.append(row[7])

        # ---
        # sentence level
        # m-index
        k = len(langs)
        total_tokens = sum(tokens_per_lang.values())
        pj_sum = 0
        for current_lang in tokens_per_lang:
            pj_sum += (tokens_per_lang[current_lang] / total_tokens) ** 2
        # i-index
        total_i_index += switches / (len(blokk) - 1)
        current_lang = None ; switches = 0
        samples.append(current_sample)
        # ---
        blokk = []
        continue

    cols = line.split("\t")
    blokk.append(cols)

# process collected stats

# m-index
k = len(langs)
total_tokens = sum(tokens_per_lang.values())
pj_sum = 0
for current_lang in tokens_per_lang:
    pj_sum += (tokens_per_lang[current_lang] / total_tokens) ** 2

m_index = (1 - pj_sum) / ((k - 1) * pj_sum)
print("M-index: ", m_index)
# ---

# language entropy
LE = 0
for current_lang in tokens_per_lang:
    pj = tokens_per_lang[current_lang] / total_tokens
    LE -= pj * math.log2(pj)

print("LE: ", LE)
# ---

# i-index
print("I-Index: ", total_i_index / number_of_sentences)
# ---

# burstiness
mean = np.mean(samples)
std = np.std(samples)
burstiness = (std - mean) / (std + mean)
print("Burstiness: ", burstiness)
# ---

# span entropy
SE = 0
spans = set(samples)
for sample in spans:
    pr = samples.count(sample) / len(samples)
    SE -= pr * math.log2(pr)
print("Span entropy: ", SE)
# ---

print(Counter(deprels_to_count))