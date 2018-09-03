import sparsevectors
import math
import pickle

# Simplest possible logger, replace with any variant of your choice.
from logger import logger
error = True     # loglevel
debug = False    # loglevel
monitor = False  # loglevel


class SemanticSpace:
    def __init__(self, dimensionality=2000, denseness=10):
        self.indexspace = {}
        self.contextspace = {}
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.permutationcollection = {}
        self.name = {}
        self.permutationcollection["nil"] = list(range(self.dimensionality))
        self.constantdenseness = 10
        self.languagemodel = LanguageModel()


    def addoperator(self, item):
        self.permutationcollection[item] = sparsevectors.createpermutation(self.dimensionality)

    def addconstant(self, item):
        self.additem(item,
                     sparsevectors.newrandomvector(self.dimensionality,
                                                   self.dimensionality // self.constantdenseness))
    def observe(self, word, loglevel=False):
        self.languagemodel.observe(word)
        if not self.contains(word):
            self.additem(word)
            logger(str(word) + " is new and now introduced: " + str(self.indexspace[word]), loglevel)

    def additem(self, item, vector="dummy"):
        if vector is "dummy":
            vector = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
        if not self.contains(item):
            self.indexspace[item] = vector
            self.contextspace[item] = sparsevectors.newemptyvector(self.dimensionality)




    def addintoitem(self, item, vector, weight=1):
        if not self.contains(item):
            vector = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
            self.indexspace[item] = vector
            self.globalfrequency[item] = 0
            self.contextspace[item] = sparsevectors.newemptyvector(self.dimensionality)
        self.contextspace[item] = \
            sparsevectors.sparseadd(self.contextspace[item], sparsevectors.normalise(vector), weight)

    def observecollocation(self, item, otheritem, operator="nil"):
        if not self.contains(item):
            self.additem(item)
        if not self.contains(otheritem):
            self.additem(otheritem)
        self.contextspace[item] = sparsevectors.sparseadd(self.contextspace[item],
                                                          sparsevectors.normalise(self.indexspace[otheritem]))
                                                      #    sparsevectors.permute(self.indexspace[otheritem],
                                                      #    self.permutationcollection[operator]))




    def outputwordspace(self, filename):
        with open(filename, 'wb') as outfile:
            for item in self.indexspace:
                try:
                    itemj = {}
                    itemj["string"] = str(item)
                    itemj["indexvector"] = self.indexspace[item]
                    itemj["contextvector"] = self.contextspace[item]
                    itemj["frequency"] = self.languagemodel.globalfrequency[item]
                    pickle.dump(itemj, outfile)
                except TypeError:
                    logger("Could not write >>" + item + "<<", error)

    def inputwordspace(self, vectorfile):
        cannedindexvectors = open(vectorfile, "rb")
        goingalong = True
        n = 0
        m = 0
        while goingalong:
            try:
                itemj = pickle.load(cannedindexvectors)
                item = itemj["string"]
                indexvector = itemj["indexvector"]
                if not self.contains(item):
                    self.additem(item, indexvector)
                    n += 1
                else:
                    self.indexspace[item] = indexvector
                    m += 1
                self.languagemodel.globalfrequency[item] = itemj["frequency"]
                self.languagemodel.bign += itemj["frequency"]  # oops should subtract previous value if any!
                self.contextspace[item] = itemj["contextvector"]
            except EOFError:
                goingalong = False
        return n, m

    def reducewordspace(self, threshold=1):
        items = list(self.indexspace.keys())
        for item in items:
            if self.languagemodel.globalfrequency[item] <= threshold:
                self.removeitem(item)

    def removeitem(self, item):
        if self.contains(item):
            del self.indexspace[item]
            del self.contextspace[item]
            self.languagemodel.bign -= self.languagemodel.globalfrequency[item]
            del self.languagemodel.globalfrequency[item]

    # ===========================================================================
    # querying the semantic space
    def contains(self, item):
        if item in self.indexspace:
            return True
        else:
            return False

    def items(self):
        return self.indexspace.keys()

    def similarity(self, item, anotheritem):
        return sparsevectors.sparsecosine(self.contextspace[item], self.contextspace[anotheritem])

    def contextneighbours(self, item, number=10):
        n = {}
        for i in self.contextspace:
            n[i] = sparsevectors.sparsecosine(self.contextspace[item], self.contextspace[i])
        return sorted(n, key=lambda k: n[k], reverse=True)[:number]

    def contextneighbourswithweights(self, item, number=10):
        n = {}
        for i in self.contextspace:
            n[i] = sparsevectors.sparsecosine(self.contextspace[item], self.contextspace[i])
        return sorted(n.items(), key=lambda k: n[k[0]], reverse=True)[:number]


    def contexttoindexneighbours(self, item, number=10):
        n = {}
        for i in self.contextspace:
            n[i] = sparsevectors.sparsecosine(self.indexspace[item], self.contextspace[i])
        return sorted(n, key=lambda k: n[k], reverse=True)[:number]

    def contexttoindexneighbourswithweights(self, item, number=10):
        n = {}
        for i in self.contextspace:
            n[i] = sparsevectors.sparsecosine(self.indexspace[item], self.contextspace[i])
        return sorted(n.items(), key=lambda k: n[k[0]], reverse=True)[:number]




    def textvector(self, words, frequencyweighting=True, binaryfrequencies=False, loglevel=False):
        self.docs += 1
        uvector = sparsevectors.newemptyvector(self.dimensionality)
        if binaryfrequencies:
            wordlist = set(words)  # not a list, a set but hey
        else:
            wordlist = words
        for w in wordlist:
            if frequencyweighting:
                factor = self.frequencyweight(w)
            else:
                factor = 1
            if w not in self.indexspace:
                self.additem(w)
            else:
                self.observe(w)
            self.df[w] += 1
            uvector = sparsevectors.sparseadd(uvector, sparsevectors.normalise(self.indexspace[w]), factor)
        return uvector

    # ===========================================================================
    # language model
    # stats associated with observed items and the collection itself
    #
    # may (actually, should) be moved to another module at some point

class LanguageModel:
    def __init__(self):
        self.globalfrequency = {}
        self.bign = 0
        self.df = {}
        self.docs = 0

    def frequencyweight(self, word, streaming=False):
        try:
            if streaming:
                l = 500
                w = math.exp(-l * self.globalfrequency[word] / self.bign)
                #
                # 1 - math.atan(self.globalfrequency[word] - 1) / (0.5 * math.pi)  # ranges between 1 and 1/3
            else:
                w = math.log((self.docs) / (self.df[word] - 0.5))
        except KeyError:
            w = 0.5
        return w


    def observe(self, word):
        self.bign += 1
        if self.contains(word):
            self.globalfrequency[word] += 1
        else:
            self.globalfrequency[word] = 1


    def contains(self, item):
        if item in self.globalfrequency:
            return True
        else:
            return False


    def importstats(self, wordstatsfile):
        with open(wordstatsfile) as savedstats:
            i = 0
            for line in savedstats:
                i += 1
                try:
                    seqstats = line.rstrip().split("\t")
                    if not self.contains(seqstats[0]):
                        self.additem(seqstats[0])
                    self.globalfrequency[seqstats[0]] = int(seqstats[1])
                    self.bign += int(seqstats[1])
                except IndexError:
                    logger("***" + str(i) + " " + line.rstrip(), debug)
