import sys
from collections import Counter

total, blokk = [], []
out = {}
full = {}

for line in sys.stdin:
	if line[0] == '#':
		continue

	if not line.strip():
		total.append(blokk)
		continue

	cols = line.rstrip("\n").split("\t")
		
	head, deprel, _, lang = cols[6:10]
	blokk.append((int(head), deprel, lang))

for blokk in total:
	for (head, deprel, lang) in blokk:
		if head == 0:
			continue

		if (blokk[head - 1][2] == 'Language=hi' and lang == 'Language=en') or (blokk[head - 1][2] == 'Language=en' and lang == 'Language=hi'):
			try:
				out[deprel] += 1
			except:
				out[deprel] = 1
		try:
			full[deprel] += 1
		except:
			full[deprel] = 1

out = Counter(out)
full = Counter(full)

for key in out:
	out[key] /= full[key]

print(Counter(out))
