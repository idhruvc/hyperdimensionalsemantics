from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import pos_tag
from logger import logger
from nltk import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
# past,  3psgpresent, past part, present, base, gerund/present participle
verbposes = ["VBD", "VBZ", "VBN", "VBP", "VB", "VBG"]

def semanticparse(string, loglevel=False):
    utterances = sent_tokenize(string)
    analyses = []
    for ss in utterances:
        analyses.append(processutterance(ss, loglevel))
    return analyses

def processutterance(ss, loglevel=False):
    string = []
    deps = []
    negation = False
    adverbial = []
    mainverb = False
    verbchain = []
    tense = False
    subject = False
    mode = False
    aspect = False
    type = False  # question, indicative, imperative, subjunctive ...
    i = 1
    words = word_tokenize(ss)
    poses = pos_tag(words)
    poslemmas = []
    sentenceitems = []
    sentencepos = []
    for pp in poses:
        postag = pp[i][0].lower()
        if postag == "j":
            postag = "a"
        if postag not in ["a", "v", "n", "r"]:
            postag = "n"  # stupid default
        lemma = lemmatizer.lemmatize(pp[0], postag)
        poslemma = (pp[0], pp[1], lemma )
        poslemmas.append(poslemma)
        sentenceitems[i] = lemma
        sentencepos[i] = pp[1]

        i += 1
    tense = "PRESENT"
    if sentencepos[root] == "VBD":
        tense = "PAST"
    if sentencepos[root] == "VBN":
        tense = "PAST"

    for edge in ss.basicDependencies.edge:
        logger(str(edge.source) + " " + sentenceitems[edge.source] +
               " " + "-" + " " + edge.dep + " " + "->" + " " +
               str(edge.target) + " " + sentenceitems[edge.target], loglevel)
        if edge.dep == 'neg' and sentencepos[edge.source] in verbposes:
            negation = True
        elif edge.dep == 'advmod':
            if edge.source == root:
                target = "epsilon"
            else:
                target = edge.source
        elif edge.dep == 'nsubj':
            subject = edge.target
        elif edge.dep == 'auxpass':
            if sentenceitems[edge.target] == "be":
                scratch['aux'].append("be")
                mode = "PASSIVE"
        elif edge.dep == 'aux':
            if sentenceitems[edge.target] == "have":
                scratch['aux'].append("have")
            if sentenceitems[edge.target] == "do":
                scratch['aux'].append("do")
            if sentenceitems[edge.target] == "be":
                scratch['aux'].append("be")
                if sentencepos[edge.source] in verbposes:
                    tense = "PROGRESSIVE"

            if sentenceitems[edge.target] == "can":
                scratch['aux'].append("can")
            if sentenceitems[edge.target] == "could":
                scratch['aux'].append("could")
            if sentenceitems[edge.target] == "would":
                scratch['aux'].append("would")
            if sentenceitems[edge.target] == "should":
                scratch['aux'].append("should")
            if sentencepos[edge.target] == "VBD":
                tense = "PAST"
            if sentenceitems[edge.target] == "will":
                scratch['aux'].append("will")
            if sentenceitems[edge.target] == "shall":
                scratch['aux'].append("shall")
    try:
        if sentencepos[root] == "VB":
            if 'aux' in scratch:
                if "will" in scratch['aux'] or "shall" in scratch['aux']:
                    tense = "FUTURE"
    except KeyError:
        logger("tense situation in " + string, True)
    features = []
    if 'aux' in scratch:
        for aa in scratch['aux']:
            features.append(aa)
    if mode:
        features.append(mode)
    if tense:
        features.append(tense)
    if negation:
        features.append("NEGATION")
    if subject:
        if sentenceitems[subject] == "I":
            features.append("p1sgsubj")
        if sentenceitems[subject] == "we":
            features.append("p1plsubj")
        if sentenceitems[subject] == "you":
            features.append("p2subj")
        #        logger(str(features) + "\t" + str(string) + "\t" + str(deps), True)
    return features


# legacy for running question categorisation experiment (2017)
# will be taken out eventually
def semanticdepparse(string, debughere=False):
    depgraph = parser_client.annotate(string)
    utterances = []
    for ss in depgraph.sentence:
        utterances.append(depparseprocess(string, ss, debughere))
    return utterances

# legacy for running question categorisation experiment (2017)
# will be taken out eventually
def depparseprocess(string, ss, debug=False):
    negated = False
    target = "epsilon"
    adverbial = "epsilon"
    subject = "epsilon"
    verb = "epsilon"
    qu = "epsilon"
    scratch = {}
    question = {}
    logger("root: " + str(ss.basicDependencies.root), debug)
    i = 0
    for w in ss.token:
        logger(str(i) + " " + w.lemma + " " + w.pos, debug)
        i += 1
    for e in ss.basicDependencies.edge:
        logger(str(e.source) + ss.token[e.source - 1].lemma + "-" + e.dep + "->" +
               str(e.target) + ss.token[e.target - 1].lemma, debug)
    sentenceitems = {}
    sentenceitems["epsilon"] = None
    sentencepos = {}
    root = ss.basicDependencies.root[0]  # only one root for now fix this!
    qu = root
    target = root
    verb = root
    i = 1
    for w in ss.token:
        sentenceitems[i] = w.lemma
        sentencepos[i] = w.pos
        scratch[i] = False
        if w.pos == "WP":
            qu = i
        if w.pos == "WRB":
            qu = i
        i += 1
    tense = "PRESENT"
    if sentencepos[root] == "VBD":
        tense = "PAST"
    if sentencepos[root] == "VBN":
        tense = "PAST"

    for edge in ss.basicDependencies.edge:
        logger(str(edge.source) + " " + sentenceitems[edge.source] +
               " " + "-" + " " + edge.dep + " " + "->" + " " +
               str(edge.target) + " " + sentenceitems[edge.target], debug)
        if edge.dep == 'nsubj':
            logger("subject:" + str(edge.target)+sentenceitems[edge.target], True)
            subject = edge.target
        elif edge.dep == 'neg':
            negated = True
        elif edge.dep == 'advmod':
            if edge.target == qu:
                if edge.source == root:
                    target = "epsilon"
                else:
                    target = edge.source
            else:
                adverbial = edge.target
        elif edge.dep == 'cop':
            if edge.target == qu:
                target = edge.source
            else:
                adverbial = edge.target
        elif edge.dep == 'aux':
            if sentenceitems[edge.target] == "have":
                scratch['aux'] = "have"
            if sentenceitems[edge.target] == "do":
                scratch['aux'] = "do"
            if sentencepos[edge.target] == "VBD":
                tense = "PAST"
            if sentenceitems[edge.target] == "will":
                scratch['aux'] = "will"
            if sentenceitems[edge.target] == "shall":
                scratch['aux'] = "shall"
    if target == "epsilon":
        if subject != "epsilon":
            target = subject
    try:
        logger(sentenceitems[root] + " " + sentencepos[root], debug)
        if sentencepos[root] == "VB":
            if 'aux' in scratch:
                if scratch['aux'] == "will" or scratch['aux'] == "shall":
                    tense = "FUTURE"
    except KeyError:
        logger("tense situation in " + string, True)
    question["question"] = sentenceitems[qu]
    question["target"] = sentenceitems[target]
    question["verb"] = sentenceitems[verb]
    question["adverbial"] = sentenceitems[adverbial]
    question["subject"] = sentenceitems[subject]
    question["tense"] = tense
    question["negated"] = negated
    #    logger(question["question"] + " " + question["target"] + " " +
    #           question["verb"] + " " + question["adverbial"] + " " +
    # question["subject"] + " " + question["tense"] + " " +
    # question["negated"] + " " + sep="\t",debug)
    return question