import sparsevectors
import math
from logger import logger
import pickle

error = True
debug = False
monitor = False


class SemanticSpace:
    def __init__(self, dimensionality=2000, denseness=10):
        self.indexspace = {}
        self.contextspace = {}
        self.associationspace = {}
        self.textspace = {}
        self.utterancespace = {}
        self.authorspace = {}
        self.globalfrequency = {}
        self.bign = 0
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.permutationcollection = {}
        self.category = {}
        self.name = {}
        self.permutationcollection["nil"] = list(range(self.dimensionality))
        self.df = {}
        self.docs = 0

    def items(self):
        return self.indexspace.keys()

    def addoperator(self, item):
        self.permutationcollection[item] = sparsevectors.createpermutation(self.dimensionality)

    def addconstant(self, item):
        self.additem(item,
                     sparsevectors.newrandomvector(self.dimensionality,
                                                   self.dimensionality // 10))

    def contains(self, item):
        if item in self.indexspace:
            return True
        else:
            return False

    def checkwordspacelist(self, words, loglevel=False):
        for word in words:
            self.checkwordspace(word, loglevel)

    def observe(self, word, loglevel=False):
        self.bign += 1
        if self.contains(word):
            self.globalfrequency[word] += 1
        else:
            self.additem(word)
            logger(str(word) + " is new and now hallucinated: " + str(self.indexspace[word]), loglevel)

    def checkwordspace(self, item, loglevel=False):
        self.observe(item, loglevel)

    def additem(self, item, vector="dummy"):
        if vector is "dummy":
            vector = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
        if not self.contains(item):
            self.indexspace[item] = vector
            self.globalfrequency[item] = 1
            self.contextspace[item] = sparsevectors.newemptyvector(self.dimensionality)
            self.associationspace[item] = sparsevectors.newemptyvector(self.dimensionality)
            self.df[item] = 0
#            self.textspace[item] = sparsevectors.newemptyvector(self.dimensionality)
#            self.pan2018tweetclassifier[item] = sparsevectors.newemptyvector(self.dimensionality)
#            self.authorspace[item] = sparsevectors.newemptyvector(self.dimensionality)
            self.bign += 1

    def addsaveditem(self, jsonitem):
        try:
            if self.contains(jsonitem["string"]):
                logger("Conflict in adding new item--- will clobber "+jsonitem["string"], error)
            item = jsonitem["string"]
            self.indexspace[item] = jsonitem["indexvector"]
            self.globalfrequency[item] = int(jsonitem["frequency"])
            self.contextspace[item] = jsonitem["contextvector"]
            self.associationspace[item] = jsonitem["associationvector"]
            self.bign += int(jsonitem["frequency"])
        except:
            logger("Something wrong with item "+jsonitem, error)

    def frequencyweight(self, word, streaming=False):
        try:
            if streaming:
                l = 500
                w = math.exp(-l * self.globalfrequency[word] / self.bign)
                    #
                    # 1 - math.atan(self.globalfrequency[word] - 1) / (0.5 * math.pi)  # ranges between 1 and 1/3
            else:
                w =  math.log((self.docs) / (self.df[word]-0.5))
        except KeyError:
            w = 0.5
        return w

    def outputwordspace(self, filename):
        with open(filename, 'wb') as outfile:
            for item in self.indexspace:
                try:
                    itemj = {}
                    itemj["string"] = str(item)
                    itemj["indexvector"] = self.indexspace[item]
                    itemj["contextvector"] = self.contextspace[item]
                    itemj["associationvector"] = self.associationspace[item]
                    itemj["frequency"] = self.globalfrequency[item]
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
                self.globalfrequency[item] = itemj["frequency"]
                self.contextspace[item] = itemj["contextvector"]
                self.associationspace[item] = itemj["associationvector"]
            except EOFError:
                goingalong = False
        return n, m


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

    def importindexvectors(self, indexvectorfile, frequencythreshold=0):
        cannedindexvectors = open(indexvectorfile, "rb")
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
                    if self.globalfrequency[item] > frequencythreshold:
                        self.indexspace[item] = indexvector
                        m += 1
            except EOFError:
                goingalong = False
        return n, m



    def reducewordspace(self, threshold=1):
        items = list(self.indexspace.keys())
        for item in items:
            if self.globalfrequency[item] <= threshold:
                self.removeitem(item)

    def removeitem(self, item):
        if self.contains(item):
            del self.indexspace[item]
            del self.contextspace[item]
            del self.associationspace[item]
            del self.globalfrequency[item]
            self.bign -= 1

    def newemptyvector(self):
        return sparsevectors.newemptyvector(self.dimensionality)

    def similarity(self, item, anotheritem):
        #  should be based on contextspace
        return sparsevectors.sparsecosine(self.indexspace[item], self.indexspace[anotheritem])

    def contextsimilarity(self, item, anotheritem):
        return sparsevectors.sparsecosine(self.contextspace[item], self.contextspace[anotheritem])

    def addintoitem(self, item, vector, weight=1):
        if not self.contains(item):
            vector = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
            self.indexspace[item] = vector
            self.globalfrequency[item] = 0
            self.contextspace[item] = sparsevectors.newemptyvector(self.dimensionality)
            self.associationspace[item] = sparsevectors.newemptyvector(self.dimensionality)
        self.contextspace[item] = sparsevectors.sparseadd(self.contextspace[item], sparsevectors.normalise(vector), weight)

    def observecollocation(self, item, otheritem, operator="nil"):
        if not self.contains(item):
            self.additem(item)
        if not self.contains(otheritem):
            self.additem(otheritem)
        self.contextspace[item] = sparsevectors.sparseadd(self.contextspace[item],
                                                          sparsevectors.normalise(self.indexspace[otheritem]))
                                                      #    sparsevectors.permute(self.indexspace[otheritem],
                                                      #    self.permutationcollection[operator]))

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


