import sys
import random

random.seed(1337)
tokens = [{'case': [], 'mark': [], 'det': []}, {'case': [], 'mark': [], 'det': []}]
# sample from first (english), modify second (hindi)
for i in [0, 1]: 
    with open(sys.argv[i + 1], "r") as f:
        for line in f:
            if line[0] == '#' or not line.strip():
                continue

            cols = line.split("\t")
            form, deprel = cols[1], cols[7]
            if deprel in ['case', 'mark', 'det']:
                # TODO: compare this with sets
                tokens[i][deprel].append(form)

for line in sys.stdin:
    if line[0] == '#' or not line.strip():
        sys.stdout.write(line)

    else:
        cols = line.split("\t")
        deprel = cols[7]
        if deprel in ['case', 'mark', 'det']:
            flip = random.randint(0, 3) == 0
            if flip:
                form = cols[1]
                if form in tokens[1][deprel]:
                    cols[1] = random.choice(tokens[0][deprel])
                else:
                    pass

        sys.stdout.write("\t".join(cols))