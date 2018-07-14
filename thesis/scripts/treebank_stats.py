import sys
from collections import Counter

blokk = []
deprels_to_count = []
for line in sys.stdin:
    if line[0] == '#':
        continue

    if not line.rstrip():
        # do stuff here
        for row in blokk:
            head = int(row[6]) - 1
            lang = row[9]
            if head != -1:
                parent_lang = blokk[head][9]
                if parent_lang != lang:
                    deprels_to_count.append(row[7])
        # stop doing stuff here
        blokk = []
        continue

    cols = line.split("\t")
    blokk.append(cols)

# process collected stats
print(Counter(deprels_to_count))