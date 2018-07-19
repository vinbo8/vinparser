import sys
import math
from collections import Counter
import numpy as np

blokk = []
deprels_to_count = []

stats = {'m': 0, 'i': 0, 'burst': 0, 'le': 0, 'se': 0, 'mem': 0, 'cl': 0}
most_freq = {}
most_freq_d1 = {}
most_freq_d2 = {}
l1, l2 = None, None
total_sentences = 0

def safe_increment(d, key):
    try:
        d[key] += 1
    except KeyError:
        d[key] = 1

for line in sys.stdin:
    if line[0] == '#':
        continue

    if not line.rstrip():
        total_sentences += 1

        # m-index
        langs = set()
        tokens_per_lang = {}

        # language entropy
        # nah

        # i-index
        current_language = None
        switches = 0

        # time stuff
        span_current_language = None
        current_span = 0
        spans = []

        # claf
        cl_in_sent = 0

        # do stuff here
        for row in blokk:
            lang = row[9]

            # m-index
            langs.add(lang)
            try:
                tokens_per_lang[lang] += 1
            except:
                tokens_per_lang[lang] = 1
            # ---

            # i-index
            if not current_language:
                current_language = lang
            elif current_language != lang:
                current_language = lang
                switches += 1
            # ---
            
            # burstiness
            if not span_current_language:
                span_current_language = lang
                current_span += 1
            elif span_current_language == lang:
                current_span += 1
            elif span_current_language != lang:
                spans.append(current_span)
                current_span = 1
            # ---

            # other
            # head = int(row[6]) - 1
            # lang = row[9]
            # if head != -1:
            #     parent_lang = blokk[head][9]
            #     if parent_lang != lang:
            #         cl_in_sent += 1
            #         safe_increment(most_freq, row[7])
            #         if not l1 and not l2:
            #             l1 = row[9] ; l2 = blokk[head][9]

            #         if lang == l1 and parent_lang == l2:
            #             safe_increment(most_freq_d1, row[7])

            #         elif lang == l2 and parent_lang == l1:
            #             safe_increment(most_freq_d2, row[7])
                        
        # ---

        # m-index
        k = len(langs)
        total_tokens = sum(tokens_per_lang.values())
        pj_sum = 0
        for current_lang in tokens_per_lang:
            pj_sum += (tokens_per_lang[current_lang] / total_tokens) ** 2

        stats['m'] += (1 - pj_sum) / ((k - 1) * pj_sum)

        # le
        lang_ent = 0
        for current_lang in tokens_per_lang:
            pj = tokens_per_lang[current_lang] / total_tokens
            lang_ent -= pj * math.log2(pj)
        
        stats['le'] += lang_ent

        # i-index
        stats['i'] += switches / (len(blokk) - 1)

        # burstiness
        mean = np.mean(spans)
        std = np.std(spans)
        stats['burst'] += (std - mean) / (std + mean)

        # memory
        m1 = np.mean(spans[:-1])
        s1 = np.std(spans[:-1])
        m2 = np.mean(spans[1:])
        s2 = np.std(spans[1:])

        # span entropy
        span_ent = 0
        span_classes = set(spans)
        for span in span_classes:
            p = spans.count(span) / len(spans)
            span_ent -= p * math.log2(p)

        stats['se'] += span_ent

        # claf
        stats['cl'] += cl_in_sent / total_tokens

        continue

    cols = line.split("\t")
    blokk.append(cols)

# process collected stats
processed = {k: v / total_sentences for (k, v) in stats.items()}
print(processed)
# print(most_freq)
# print(l1, l2)

# most_freq = {k: (v / sum(most_freq.values())) for (k, v) in most_freq.items()}
# most_freq_d1 = {k: (v / sum(most_freq_d1.values())) for (k, v) in most_freq_d1.items()}
# most_freq_d2 = {k: (v / sum(most_freq_d2.values())) for (k, v) in most_freq_d2.items()}

# most_freq = Counter(most_freq_d2)
# for k, v in most_freq.most_common(10):
#     print("\\texttt{{{}}} & {:.2f} & ".format(k, v * 100))
# # print(Counter(most_freq_d1))
# print(Counter(most_freq_d2))
