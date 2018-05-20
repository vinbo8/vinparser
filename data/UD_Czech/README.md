# Summary

The Czech-PDT UD treebank is based on the Prague Dependency Treebank 3.0 (PDT),
created at the Charles University in Prague.


# Introduction

The treebank consists of 87,913 sentences (1.5 M tokens) and its domain is
mainly newswire, reaching also to business and popular scientific articles
from the 1990s. The treebank is licensed under the terms of
[CC BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/)
and its original (non-UD) version can be downloaded from
[http://hdl.handle.net/11858/00-097C-0000-0023-1AAF-3](http://hdl.handle.net/11858/00-097C-0000-0023-1AAF-3).

The morphological and syntactic annotation of the Czech UD treebank is created
through a conversion of PDT data. The conversion procedure has been designed by
Dan Zeman and implemented in Treex.


# Acknowledgments

We wish to thank all of the contributors to the original PDT annotation effort,
including Eduard Bejček, Eva Hajičová, Jan Hajič, Pavlína Jínová,
Václava Kettnerová, Veronika Kolářová, Marie Mikulová, Jiří Mírovský,
Anna Nedoluzhko, Jarmila Panevová, Lucie Poláková, Magda Ševčíková,
Jan Štěpánek, and Šárka Zikánová.

## References

* Eduard Bejček, Eva Hajičová, Jan Hajič, Pavlína Jínová, Václava Kettnerová,
  Veronika Kolářová, Marie Mikulová, Jiří Mírovský, Anna Nedoluzhko,
  Jarmila Panevová, Lucie Poláková, Magda Ševčíková, Jan Štěpánek,
  and Šárka Zikánová. 2013. Prague Dependency Treebank 3.0,
  LINDAT/CLARIN digital library at Institute of Formal and Applied Linguistics,
  Charles University in Prague,
  http://hdl.handle.net/11858/00-097C-0000-0023-1AAF-3.

* Eduard Bejček, Jarmila Panevová, Jan Popelka, Pavel Straňák, Magda Ševčíková,
  Jan Štěpánek, and Zdeněk Žabokrtský. 2012. Prague Dependency Treebank 2.5 –
  a revisited version of PDT 2.0.
  In: Proceedings of the 24th International Conference on Computational
  Linguistics (Coling 2012), Mumbai, India, pp. 231-246.
  http://www.aclweb.org/anthology/C/C12/C12-1015.pdf


# Domains and Data Split

NOTE: Earlier releases of the treebank had four training data files. This was
due to Github restrictions on file size. We have now re-joined the training
files in the official release package (beginning with UD v1.3), so there is
just one training file as in all other languages, and it is named
cs-ud-train.conllu. The four files in previous releases corresponded to the
four sources of the original texts; the sources may still be distinguished,
if desirable, by the prefixes of sentence ids. All of them are newspapers, but

* l (ln) and m (mf) are mainstream daily papers (news, commentaries, but also
  sports results and TV programs)
* c (cmpr) is a business weekly
* v (vesm) contains popular scientific articles (the hardest to parse: long
  sentences and unusual vocabulary)

The dev and test sets contain all four sources and their size is proportional
to the sizes of the respective training parts.


## Source of annotations

This table summarizes the origins and checking of the various columns of the CoNLL-U data.

| Column | Status |
| ------ | ------ |
| ID | Sentence segmentation and (surface) tokenization was automatically done and then hand-corrected; see [PDT documentation](http://ufal.mff.cuni.cz/pdt2.0/doc/pdt-guide/en/html/ch02.html). Splitting of fused tokens into syntactic words was done automatically during PDT-to-UD conversion. |
| FORM | Identical to Prague Dependency Treebank 3.0 form. |
| LEMMA | Manual selection from possibilities provided by morphological analysis: two annotators and then an arbiter. PDT-to-UD conversion stripped from lemmas the ID numbers distinguishing homonyms, semantic tags and comments; this information is preserved as attributes in the MISC column. |
| UPOSTAG | Converted automatically from XPOSTAG (via [Interset](https://ufal.mff.cuni.cz/interset)), from the semantic tags in PDT lemma, and occasionally from other information available in the treebank; human checking of patterns revealed by automatic consistency tests. |
| XPOSTAG | Manual selection from possibilities provided by morphological analysis: two annotators and then an arbiter. |
| FEATS | Converted automatically from XPOSTAG (via Interset), from the semantic tags in PDT lemma, and occasionally from other information available in the treebank; human checking of patterns revealed by automatic consistency tests. |
| HEAD | Original PDT annotation is manual, done by two independent annotators and then an arbiter. Automatic conversion to UD; human checking of patterns revealed by automatic consistency tests. |
| DEPREL | Original PDT annotation is manual, done by two independent annotators and then an arbiter. Automatic conversion to UD; human checking of patterns revealed by automatic consistency tests. |
| DEPS | &mdash; (currently unused) |
| MISC | Information about token spacing taken from PDT annotation. Lemma / word sense IDs, semantic tags and comments on meaning moved here from the PDT lemma. |


# Changelog

* 2018-04-15 v2.2
  * Repository renamed from UD_Czech to UD_Czech-PDT.
  * Added enhanced representation of dependencies propagated across coordination.
    The distinction of shared and private dependents is derived deterministically from the original Prague annotation.
  * Fixed computation of the LDeriv MISC attribute.
* 2017-11-15 v2.1
  * Retagged pronouns “každý” and “kterýžto”.
  * Prepositional objects are now “obl:arg” instead of “obj”.
  * Instrumental phrases for demoted agents in passives are now “obl:agent”.
* 2017-03-01 v2.0
  * Converted to UD v2 guidelines.
  * Reconsidered PRON vs. DET. Extended PronType and Poss.
  * Improved advmod vs. obl distinction.
  * L-participles are verbs, other participles are adjectives.
  * Removed style flags from lemmas.
* 2016-05-15 v1.3
  * Fixed adverbs that were attached as nmod; correct: advmod.
  * Copulas with clausal complements are now heads.
  * Improved conversion of AuxY.
  * Relation of foreign prepositions changed to foreign.
* 2015-11-15 v1.2
  * Conversion procedure rewritten again (may result in minor differences in
    borderline cases)
  * Only one “root” relation per tree now enforced; some bugs around root fixed
  * The “name” relation goes now always left-to-right (in UD 1.1 it was family-
    to-given name)
  * Fixed bug with numeral-noun swapping that destroyed coordinations of
    numbers and caused the “conj” relation to go right-to-left
  * Fixed minor bugs around subordinating conjunctions
  * Changed dependency relation of reflexive pronouns attached to inherently
    reflexive verbs from compound:reflex to expl
  * Applied heuristics to distinguish at least some iobj from dobj
  * Fixed bugs around xcomp (future infinitives and subjects attached to
    controlled verbs)
* 2015-05-15 v1.1
  * Conversion procedure completely rewritten
  * Improved heuristics to distinguish DET and PRON
  * Improved treatment of comparative complements (conjunctions “než” and “jako”)
  * Remaining lemma extensions moved from LEMMA to MISC



<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v1.0
License: CC BY-NC-SA 3.0
Includes text: yes
Genre: news reviews nonfiction
Lemmas: converted from manual
UPOS: converted from manual
XPOS: manual native
Features: converted from manual
Relations: converted from manual
Contributors: Zeman, Daniel; Hajič, Jan
Contributing: elsewhere
Contact: zeman@ufal.mff.cuni.cz
===============================================================================
</pre>
