from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import re
import semanticdependencyparse
import lexicalfeatures.lexicon as lexicon
from logger import logger

urlpatternexpression = re.compile(r"https?://[/A-Za-z0-9\.\-\?_]+", re.IGNORECASE)
handlepattern = re.compile(r"@[A-Za-z0-9_\-±.]+", re.IGNORECASE)
verbtags = ["VB", "VBZ", "VBP", "VBN", "VBD", "VBG"]
adjectivetags = ["JJ", "JJR", "JJS"]
def words(text):
    words = word_tokenize(text)
    return words


def generalise(text, handlesandurls=True, nouns=True, verbs=True, adjectives=True, adverbs=False):
    accumulator = []
    if handlesandurls:
        text = urlpatternexpression.sub("U", text)
        text = handlepattern.sub("H", text)
    sents = sent_tokenize(text)
    for sentence in sents:
        words = word_tokenize(sentence)
        poses = pos_tag(words)
        for item in poses:
            if nouns and item[1] == "NN":
                accumulator.append("N")
            elif nouns and item[1] == "NNS":
                accumulator.append("Ns")
            elif adjectives and item[1] in adjectivetags:
                accumulator.append(item[1])
            elif verbs and item[1] in verbtags:
                tag = item[1]
                if tag == "VBZ":
                    tag = "VBP"  #  neutralise for 3d present -- VBP is present
                accumulator.append(tag)
            elif adverbs and item[1] == "RB":
                accumulator.append("R")
            else:
                accumulator.append(item[0])
    return " ".join(accumulator)


# do  MD (modal) (separate out 'not' from RB)


def featurise(text, loglevel=False):
    features = []
    sents = sent_tokenize(text)
    for sentence in sents:
        words = word_tokenize(sentence)
        for word in words:
            for feature in lexicon:
                if word.lower() in lexicon[feature]:
                    features.append("JiK" + feature)
        parsedfeatures = semanticdependencyparse.semanticdependencyparse(text)
        for featureset in parsedfeatures:
            features += featureset
    logger(text + "->" + str(features), loglevel)
    return features



goodgenderones = ["JiKlove", "JiKp1", "JiKp1sgsubj",
                  "JiKamplifySurprise", "JiKinterjection", "JiKp2", "JiKPROGRESSIVE", "JiKPAST", "JiKshould",
                  "JiKbe", "JiKwill", "JiKinsecure", "JiKplaceadverbial", "JiKthinkverbs", "JiKPASSIVE",
                  "JiKsayverbs", "JiKprofanity", "JiKwould", "JiKhedgelist", "JiKexciting", "JiKembarrassing",
                  "JiKthank", "JiKwomen", "JiKfamily"]


mediocremale = ["JiKp1plsubj", "JiKmotionverbs",
                                       "JiKsurprise", "JiKVERYLATEMAINV", "JiKcould", "JiKdislike", "JiKhave",
                                       "JiKFUTURE", "JiKboredom", "JiKp3", "JiKEARLYMAINV"]
mediocrefemale = ["JiKcan", "JiKp2subj", "JiKNEGATION",
                                       "JiKamplifyTruly", "JiKhate", "JiKamplifier", "JiKpositive"]

mediocregenderones = goodgenderones + mediocrefemale + mediocremale

def mildpositems(string, full=False):
    leaveintags = ["IN", "DT", "MD", "PRP", "PRP$", "POS", "CC", "EX", "PDT", "RP", "TO", "WP", "WP$", "WDT", "WRB"]
    words = word_tokenize(string)
    poses = pos_tag(words)
    if not full:
        returnposes = [("START", "START")]
        for p in poses:
            if p[1] in leaveintags:
                returnposes.append((p[1],p[0]))
            else:
                returnposes.append(p)
        returnposes.append(("END","END"))
    else:
        returnposes = [("START", "BEG")] + poses + [("END", "END")]
    return returnposes
