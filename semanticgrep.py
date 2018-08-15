import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from logger import logger
import sparsevectors
import semanticdependencyparse
import hyperdimensionalsemanticspace


# take a file with an utterance per line
# process it one by one,
# - parse it
# - generate lexical vectors
# - never mind context vectors for now
# -
# return a vector per utterance / line

sentencestorage = {}
utterancespace = {}
textspace = {}
wordspace = hyperdimensionalsemanticspace.SemanticSpace()

debug = False
monitor = True
error = True

def processfile(file):
    global sentencestorage, utterancespace
    sentenceindex = 0
    textvector = wordspace.newemptyvector()
    with open(file, "r", encoding="utf-8") as textfile:
        rawtext = textfile.read().lower()
        rawtext = re.sub('\n', ' ', rawtext)
        rawtext = re.sub('\"', ' ', rawtext)
        rawtext = re.sub('\s+', ' ', rawtext)
        sents = sent_tokenize(rawtext)
        for sentence in sents:
            sentenceindex += 1
            sentencestorage[sentenceindex] = sentence
            allsurfacewords = nltk.word_tokenize(sentence)
            wordspace.chkwordspace(allsurfacewords, debug)
            analyses = []
            try:
                analyses = semanticdependencyparse.semanticdepparse(sentence.lower(), debug)
            except:
                logger("PARSE ERROR " + str(sentenceindex) + "\t" + sentence, error)
            kk = 0
            for analysis in analyses:
                words = analysis.values()
                wordspace.checkwordspacelist(words, debug)
                for role in analysis:
                    if role not in wordspace.permutationcollection:
                        wordspace.permutationcollection[role] = sparsevectors.createpermutation(wordspace.dimensionality)
                u = getvector(analysis, sentence)
                win = 1
                sentencesequence = 0
                startindexforthistext = 0
                while win < sentencesequence:
                    if sentenceindex - win > startindexforthistext:
                        u = sparsevectors.sparseadd(u, sparsevectors.permute(
                            sparsevectors.normalise(utterancespace[sentenceindex - win]),
                            wordspace.permutationcollection["discourse"]))
                    win += 1
                if kk > 0:
                    sentenceindex += 1
                utterancespace[sentenceindex] = u
                textvector = sparsevectors.sparseadd(textvector, u, 1)
                kk += 1
        textspace[file] = textvector
    return textvector


amplifyGrade = ["very", "awfully", "completely", "enormously", "entirely", "exceedingly", "excessively", "extremely",
                "greatly", "highly", "hugely", "immensely", "intensely", "particularly", "radically", "significantly",
                "strongly", "substantially", "totally", "utterly", "vastly"]

amplifyTruly = ["absolutely", "definitely", "famously", "genuinely", "immaculately", "overly", "perfectly", "really",
                "severely", "surely", "thoroughly", "truly", "undoubtedly"]

amplifySurprise = ["amazingly", "dramatically", "drastically", "emphatically", "exceptionally", "extraordinarily",
                   "fantastically", "horribly", "incredibly", "insanely", "phenomenally", "remarkably", "ridiculously",
                   "strikingly", "surprisingly", "terribly", "unusually", "wildly", "wonderfully"]

negationlist = ["no", "none", "never", "not", "n't", "neither", "nor"]
amplifierlist = amplifyGrade + amplifySurprise + amplifyTruly
hedgelist = ["apparently", "appear", "around", "basically", "effectively", "evidently", "fairly", "generally",
             "hopefully", "largely", "likely", "mainly", "maybe", "mostly", "overall", "perhaps", "presumably",
             "probably", "quite", "rather", "somewhat", "supposedly", "possibly", "doubtfully", "arguably", "often",
             "unlikely", "usually", "sometimes", "certainly", "definitely", "clearly", "conceivably", "apparent",
             "certain", "possible", "presumed", "probable", "putative", "supposed", "doubtful", "appear", "assume",
             "estimate", "indicate", "infer", "intend", "presume", "propose", "seem", "speculate", "suggest", "suppose",
             "tend", "doubt"]


def getvector(roleworddict, sentencestring):
    uvector = {}  # vector for test item
    for role in roleworddict:
        item = roleworddict[role]
        uvector = sparsevectors.sparseadd(uvector,
                                          sparsevectors.permute(sparsevectors.normalise(wordspace.indexspace[item]),
                                                                wordspace.permutationcollection[role]),
                                          wordspace.frequencyweight(item))
    lexicalwindow = 1
    if lexicalwindow > 0:
        wds = word_tokenize(sentencestring.lower())
        windows = [wds[i:i + lexicalwindow] for i in range(len(wds) - lexicalwindow + 1)]
        for sequence in windows:
            thisvector = {}
            for item in sequence:
                thisvector = sparsevectors.sparseadd(
                    sparsevectors.permute(thisvector, wordspace.permutationcollection["sequence"]),
                    wordspace.indexspace[item],
                    wordspace.frequencyweight(item))
            uvector = sparsevectors.sparseadd(uvector, sparsevectors.normalise(thisvector))
    pos = 1
    if pos > 0:
        wds = word_tokenize(sentencestring)
        posanalyses = nltk.pos_tag(wds)
        poslist = [i[1] for i in posanalyses]
        windows = [poslist[i:i + lexicalwindow] for i in range(len(poslist) - lexicalwindow + 1)]
        for sequence in windows:
            thisvector = {}
            for item in sequence:
                thisvector = sparsevectors.sparseadd(
                    sparsevectors.permute(thisvector, wordspace.permutationcollection["sequence"]),
                    wordspace.indexspace[item],
                    wordspace.frequencyweight(item))
            uvector = sparsevectors.sparseadd(uvector, sparsevectors.normalise(thisvector))
    style = True
    if style:
        wds = word_tokenize(sentencestring)
        cpw = len(sentencestring) / len(wds)
        wps = len(wds)
        sl = True
        if sl:
            if wps > 8:
                uvector = sparsevectors.sparseadd(uvector, longsentencevector)
            if wps < 5:
                uvector = sparsevectors.sparseadd(uvector, shortsentencevector)
        posanalyses = nltk.pos_tag(wds)
        poslist = [i[1] for i in posanalyses]
        for poses in poslist:
            if poses == "RB" or poses == "RBR" or poses == "RBS":
                uvector = sparsevectors.sparseadd(uvector, adverbvector)
        for w in wds:
            if w in negationlist:
                uvector = sparsevectors.sparseadd(uvector, negationvector)
            if w in hedgelist:
                uvector = sparsevectors.sparseadd(uvector, hedgevector)
            if w in amplifierlist:
                uvector = sparsevectors.sparseadd(uvector, amplifiervector)

    # attitude terms
    # verb stats
    # seq newordgrams
    # verb classes use wordspace!
    # sent sequences
    return uvector



def main():
    runtest()

if __name__ == "__main__":
    main()
