import hyperdimensionalsemanticspace
import sparsevectors

number = 10
negattitudewordset = set()
posattitudewordset = set()


vecs = hyperdimensionalsemanticspace.SemanticSpace()
wordspacedirectory = "/home/jussi/data/wordspaces/"
wordspacefile = "canonical.space.2017-09-05.EN.twitter.jq.irma"
apfile = "canonical.space.2.ap"
vecs.inputwordspace(wordspacedirectory + wordspacefile)

with open("/home/jussi/data/poles/en/enposBingLiu.list", "r") as posfile:
    line = posfile.readline()
    lineno = 0
    while line:
        lineno += 1
        word = line.rstrip()
        posattitudewordset.add(word)
        line = posfile.readline()

with open("/home/jussi/data/poles/en/ennegBingLiu.list", "r") as negfile:
    line = negfile.readline()
    lineno = 0
    while line:
        lineno += 1
        word = line.rstrip()
        negattitudewordset.add(word)
        line = negfile.readline()



for i in vecs.indexspace:
    p = 0
    n = 0
    negscore = "--"
    ampSscore = "--"
    ampTscore = "--"
    ampGscore = "--"
    dtscore = "--"
    wereon = False
    if vecs.globalfrequency[i] > 1:
        wereon = True
        ns = dict(vecs.contextneighbourswithweights(i, number))
        negscore = sparsevectors.sparsecosine(vecs.indexspace["JiKnegation"], vecs.contextspace[i])
        ampSscore = sparsevectors.sparsecosine(vecs.indexspace["JiKampsurprise"], vecs.contextspace[i])
        ampTscore = sparsevectors.sparsecosine(vecs.indexspace["JiKamptruly"], vecs.contextspace[i])
        ampGscore = sparsevectors.sparsecosine(vecs.indexspace["JiKampgrade"], vecs.contextspace[i])
        dtscore = sparsevectors.sparsecosine(vecs.indexspace["JiKhedge"], vecs.contextspace[i])
    if str(i).startswith("JiK"):
        ns = {}
        wereon = True
        for j in vecs.contextspace:
            ns[j] = sparsevectors.sparsecosine(vecs.indexspace[i], vecs.contextspace[j])
    if wereon:
        k = sorted(ns.items(), key=lambda k: ns[k[0]], reverse=True)[:number]
        for witem in k:
            if witem[0] in posattitudewordset:
                p += 1
            if witem[0] in negattitudewordset:
                n += 1
        if n + p > 0:
            r = p / (n + p)
        else:
            r = 0
        print(i, vecs.globalfrequency[i], p, n, r, negscore, ampSscore, ampTscore, ampGscore, dtscore, sep="\t")
        print(k)