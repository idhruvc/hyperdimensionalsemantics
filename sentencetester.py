import os

import nltk

os.environ["CORENLP_HOME"] = "/usr/share/stanford-corenlp-full/"


from hyperdimensionalsemanticspace import SemanticSpace
import stringsequencespace
import sparsevectors
from logger import logger
import xml.etree.ElementTree
import squintinglinguist

"""
This program builds an pan2018tweetclassifier from a file and tests it against some given sentences.
"""

dimensionality = 2000
denseness = 10
window = 0

debug = False
monitor = True
error = True

# frequency statistics for observable items
wordstatsfile = "/home/jussi/data/wordspaces/pan18.wordstats"
# True or false
frequencyweighting = True
charactervectorspacefilename = "/home/jussi/data/wordspaces/factory.characters.author.weight.fgp"

stringspace = stringsequencespace.StringSequenceSpace(dimensionality, denseness, window)
stringspace.importelementspace(charactervectorspacefilename)

textfile = "/home/jussi/data/pan/pan18-author-struct.xml"

logger("Reading frequencies from " + wordstatsfile, monitor)
stringspace.importstats(wordstatsfile)

logger("Text target space", monitor)

logger("Started training.", monitor)
textindex = 0

textspace = SemanticSpace(dimensionality, denseness)
modifiedtextspace = SemanticSpace(dimensionality, denseness)
squintfeaturespace = SemanticSpace(dimensionality, denseness)
fullspace = SemanticSpace(dimensionality, denseness)

textdepot = {}
modifiedtextdepot = {}
featuredepot = {}

e = xml.etree.ElementTree.parse(textfile).getroot()
for b in e.iter("document"):
    textindex += 1
    tvector = sparsevectors.normalise(stringspace.textvector(b.text, frequencyweighting))
    textspace.additem(textindex, tvector)
    newtext = squintinglinguist.generalise(b.text)
    mvector = sparsevectors.normalise(stringspace.textvector(newtext, frequencyweighting))
    modifiedtextspace.additem(textindex, mvector)
    features = squintinglinguist.featurise(b.text)
    fvector = sparsevectors.newemptyvector(dimensionality)
    for feature in features:
        fv = stringspace.getvector(feature)
        fvector = sparsevectors.sparseadd(fvector, sparsevectors.normalise(fv), stringspace.frequencyweight(feature))
    fvector = sparsevectors.normalise(fvector)
    squintfeaturespace.additem(textindex, fvector)
    pvector = sparsevectors.normalise(stringspace.postriplevector(b.text))
    avector = sparsevectors.sparseadd(pvector, sparsevectors.sparseadd(mvector,
                                                                       sparsevectors.sparseadd(fvector, tvector)))
    fullspace.additem(textindex, avector)
    textdepot[textindex] = b.text
    modifiedtextdepot[textindex] = newtext
    featuredepot[textindex] = features
logger("Done making " + str(textindex) + " vectors.", monitor)

matrix = False
if matrix:
    for space in [textspace, modifiedtextspace, squintfeaturespace, fullspace]:
        logger("neighbour calc", monitor)
        ffneighbours = {}
        for item in space.items():
            ffneighbours[item] = {}
            for otheritem in space.items():
                ffneighbours[item][otheritem] = space.similarity(item, otheritem)
        print("\t", end="\t")
        for item in space.items():
            print(item, end="\t")
        print()
        for item in space.items():
            print(item, end="\t")
            for otheritem in space.items():
                print(ffneighbours[item][otheritem], end="\t")
            print()

for newtest in textdepot:
    origtext = textdepot[newtest]
    tvector = sparsevectors.normalise(stringspace.textvector(origtext, frequencyweighting))
    newtext = squintinglinguist.generalise(newtext)
    mvector = sparsevectors.normalise(stringspace.textvector(newtext, frequencyweighting))
    features = squintinglinguist.featurise(origtext)
    fvector = sparsevectors.newemptyvector(dimensionality)
    for feature in features:
        fv = stringspace.getvector(feature)
        fvector = sparsevectors.sparseadd(fvector, sparsevectors.normalise(fv), stringspace.frequencyweight(feature))
    fvector = sparsevectors.normalise(fvector)
    pvector = sparsevectors.normalise(stringspace.postriplevector(origtext))
    avector = sparsevectors.sparseadd(pvector, sparsevectors.sparseadd(mvector,
                                                                       sparsevectors.sparseadd(fvector, tvector)))
    vector = fvector
    tn = {}
    mn = {}
    fn = {}
    an = {}
    nofn = 3
    for otheritem in fullspace.items():
        if otheritem == newtest:
            continue
        tn[otheritem] = sparsevectors.sparsecosine(tvector, fullspace.indexspace[otheritem])
        mn[otheritem] = sparsevectors.sparsecosine(mvector, fullspace.indexspace[otheritem])
        fn[otheritem] = sparsevectors.sparsecosine(fvector, fullspace.indexspace[otheritem])
        an[otheritem] = sparsevectors.sparsecosine(avector, fullspace.indexspace[otheritem])
    logger(str(newtest) + "\t" + textdepot[newtest], debug)
    tnn = sorted(tn, key=lambda i:tn[i], reverse=True)[:nofn]
    logger(str(tnn), debug)
    for o in tnn:
        logger("\t" + str(o) + "\t" + str(tn[o]) + "\t" + textdepot[o], debug)
    mnn = sorted(mn,key=lambda i:mn[i], reverse=True)[:nofn]
    logger(str(mnn), debug)
    for o in mnn:
        logger("\t" + str(o) + "\t" + str(mn[o]) + "\t" + textdepot[o], debug)
    fnn = sorted(fn, key=lambda i:fn[i], reverse=True)[:nofn]
    logger(str(fnn), debug)
    for o in fnn:
        logger("\t" + str(o) + "\t" + str(fn[o]) + "\t" + textdepot[o], debug)
    ann = sorted(an, key=lambda i:an[i], reverse=True)[:nofn]
    logger(str(ann), debug)
    for o in ann:
        logger("\t" + str(o) + "\t" + textdepot[o], debug)

sampleitems = ["I really did not like the clarinet, I am afraid: it sounded weak!"]

for probetext in sampleitems:
    tvector = sparsevectors.normalise(stringspace.textvector(probetext, frequencyweighting))
    newtext = squintinglinguist.generalise(probetext)
    mvector = sparsevectors.normalise(stringspace.textvector(newtext, frequencyweighting))
    features = squintinglinguist.featurise(probetext)
    fvector = sparsevectors.newemptyvector(dimensionality)
    for feature in features:
        fv = stringspace.getvector(feature)
        fvector = sparsevectors.sparseadd(fvector, sparsevectors.normalise(fv), stringspace.frequencyweight(feature))
    fvector = sparsevectors.normalise(fvector)
    pvector = sparsevectors.normalise(stringspace.postriplevector(probetext))
    avector = sparsevectors.sparseadd(sparsevectors.normalise(sparsevectors.sparseadd(mvector, pvector)),
                                      sparsevectors.normalise(sparsevectors.sparseadd(fvector, tvector)))
    vector = fvector
    tn = {}
    mn = {}
    fn = {}
    pn = {}
    an = {}
    nofn = 3
    for otheritem in fullspace.items():
        tn[otheritem] = sparsevectors.sparsecosine(tvector, fullspace.indexspace[otheritem])
        mn[otheritem] = sparsevectors.sparsecosine(mvector, fullspace.indexspace[otheritem])
        fn[otheritem] = sparsevectors.sparsecosine(fvector, fullspace.indexspace[otheritem])
        pn[otheritem] = sparsevectors.sparsecosine(pvector, fullspace.indexspace[otheritem])
        an[otheritem] = sparsevectors.sparsecosine(avector, fullspace.indexspace[otheritem])
    tnn = sorted(tn, key=lambda i:tn[i], reverse=True)[:nofn]
    logger(str(tnn), monitor)
    logger("\ttext:\t" + probetext, monitor)
    for o in tnn:
        logger("\t" + str(o) + "\t" + str(tn[o]) + "\t" + textdepot[o], monitor)
        wds = nltk.word_tokenize(textdepot[o])
        www = {}
        for w in wds:
            www[w] = sparsevectors.sparsecosine(stringspace.indexspace[w], tvector)
        wwwn = sorted(www.items(), key=lambda i: www[i[0]], reverse=True)[:nofn]
        logger("\t\t" + wwwn, monitor)

    mnn = sorted(mn, key=lambda i:mn[i], reverse=True)[:nofn]
    logger(str(mnn), monitor)
    logger("\tgentext:\t" + newtext, monitor)
    for o in mnn:
        logger("\t" + str(o) + "\t" + str(mn[o]) + "\t" + textdepot[o], monitor)
    fnn = sorted(fn, key=lambda i:fn[i], reverse=True)[:nofn]
    logger(str(fnn), monitor)
    logger("\tfeatures:\t" + str(features), monitor)
    for o in fnn:
        logger("\t" + str(o) + "\t" + str(fn[o]) + "\t" + textdepot[o], monitor)
    pnn = sorted(pn, key=lambda i:pn[i], reverse=True)[:nofn]
    logger(str(pnn), monitor)
    for o in pnn:
        logger("\t" + str(o) + "\t" + str(pn[o]) + "\t" + textdepot[o], monitor)
    ann = sorted(an, key=lambda i:an[i], reverse=True)[:nofn]
    logger(str(ann), monitor)
    for o in ann:
        logger("\t" + str(o) + "\t" + str(an[o]) + "\t" + textdepot[o], monitor)


