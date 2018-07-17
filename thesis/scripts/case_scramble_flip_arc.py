import sys
import random

random.seed(1337)
tokens = [{'case': [], 'mark': [], 'det': []}, {'case': [], 'mark': [], 'det': []}]
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

blokk, edited_in_sentence, offset = [], [], 0
for line in sys.stdin:
    if not line.rstrip():
        for edited_id in edited_in_sentence:
            head = int(blokk[edited_id][6]) - 1

            # swap
            temp = blokk[head]
            blokk[head] = blokk[edited_id]
            blokk[edited_id] = temp

            # fix IDs
            new_id = blokk[head][0]
            old_id = blokk[edited_id][0]
            blokk[head][0] = blokk[edited_id][0]
            blokk[edited_id][0] = new_id

            # fix other rels
            for blokk_line in blokk:
                if blokk_line[6] == new_id:
                    blokk_line[6] = old_id
                elif blokk_line[6] == old_id:
                    blokk_line[6] = new_id

        for blokk_line in blokk:
            try:
                sys.stdout.write("\t".join(blokk_line))
            except:
                print(blokk_line)
                sys.exit()
        blokk, edited_in_sentence, offset = [], [], 0
        sys.stdout.write(line)

    elif line[0] == '#':
        sys.stdout.write(line)

    else:
        cols = line.split("\t")
        if "." in cols[0] or "-" in cols[0]:
            offset += 1

        deprel = cols[7]
        if deprel in ['case', 'mark', 'det']:
            flip = random.randint(0, 0) == 0
            if flip:
                # subtract one to get index in blokk
                form, head = cols[1], (int(cols[6]) - 1)
                if form in tokens[1][deprel]:
                    cols[1] = random.choice(tokens[0][deprel])
                    edited_in_sentence.append(int(cols[0]) - 1 + offset)
                else:
                    pass

        blokk.append(cols)
#        sys.stdout.write("\t".join(cols))