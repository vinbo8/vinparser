import sys
from collections import Counter

blokker = []
blokk = []
deprels_to_count = []
n = 0
for line in sys.stdin:
    n += 1
    if line[0] == '#':
        sys.stdout.write(line)
        continue

    if not line.rstrip():
        # do stuff here
        current_span = None
        for row in blokk:
            lang = row[9]
            if lang == "Language=en\n" or lang == "Language=hi\n":
                current_span = lang
            
            else:
                if current_span:
                    row[9] = current_span

                else:
                    for seeker in blokk:
                        lang = seeker[9]
                        if lang == "Language=en\n" or lang == "Language=hi\n":
                            row[9] = lang
                            break

        
            sys.stdout.write("\t".join(row))	

        sys.stdout.write("\n")
        # stop doing stuff here
        blokk = []
        continue

    cols = line.split("\t")
    blokk.append(cols)
