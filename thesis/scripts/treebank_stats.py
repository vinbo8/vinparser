import sys
import math
from collections import Counter
import numpy as np

blokk = []
deprels_to_count = []

stats = {'m': 0, 'i': 0, 'burst': 0, 'le': 0, 'se': 0, 'mem': 0}
total_sentences = 0

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
            head = int(row[6]) - 1
            lang = row[9]
            if head != -1:
                parent_lang = blokk[head][9]
                if parent_lang != lang:
                    deprels_to_count.append(row[7])

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
        continue

    cols = line.split("\t")
    blokk.append(cols)

# process collected stats
processed = {k: v / total_sentences for (k, v) in stats.items()}
print(processed)