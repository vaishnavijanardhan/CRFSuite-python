from hw3_corpus_tool import *
import os.path
import glob
import pycrfsuite
import sys

def calculateAccuracy(predLabel, trueLabel):
    count = 0
    accuracy = 0
    for i in range(len(predLabel)):
        temp11 = len(predLabel[i])
        for j in range(temp11):
            count = count + 1
            if predLabel[i][j] == trueLabel[i][j]:
                accuracy = 1 + accuracy
    accuracy = float(accuracy) / float(count)
    print("Accuracy : {} ".format(accuracy))
    return accuracy

def test(test_dir, funcFeature, write_file):
    trueLabel = []
    predLabel = []
    output = open(write_file, 'w')
    tagger = pycrfsuite.Tagger()
    tagger.open("model.crfsuite")
    dialog_filenames = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

    for dialog_filename in dialog_filenames:
        dialog = get_utterances_from_filename(dialog_filename)
        xseq = funcFeature(dialog)
        predLabel.append(tagger.tag(xseq))
        temp = convertToTag(dialog)
        trueLabel.append(temp)

    for index, dialog_filename in enumerate(dialog_filenames):
        output.write("Filename="+os.path.basename(dialog_filename))
        for label in predLabel[index]:
            output.write(label + "\n")
        output.write("\n")
    output.close()

    if (not(None in trueLabel)):
        return calculateAccuracy(predLabel, trueLabel)

def convertToTag(dialog):
    x = [utt.act_tag for utt in dialog]
    return x

def func_baseline(dialog):
    features = []
    for index, utt in enumerate(dialog):
        feature = []
        if (index == 0):
            feature.append("FirstUtt")
        if (dialog[index].speaker != dialog[index-1].speaker and index > 0):
            feature.append("Speaker_Changed")
        if utt.pos:
            feature = feature + ["PartOfSpeech" + word.pos for word in utt.pos]
        if utt.pos:
            feature = feature + ["Token" + word.token for word in utt.pos]
        features.append(feature)
    return features

def func_advanced(dialog):
    features = []
    for index, utt in enumerate(dialog):
        feature = {}
        if index > 0 and dialog[index].speaker != dialog[index-1].speaker:
            feature["Speaker_Changed"] = 1
        if index == 0:
            feature["FirstUtt"] = 1
        if (utt.pos):
            tokens = [word.token for word in utt.pos]
            feature['Token'] = tokens
            lis_t = [word.pos for word in utt.pos]
            feature['PartOfSpeech'] = lis_t
            leng_t = len(utt.pos)
            feature['Length'] = leng_t
            if(utt.pos[-1].token == '?'):
                feature['Statement'] = 'Question'
            else:
                feature['Statement'] = 'Answer'
            bigrams = list(zip(tokens[:-1], tokens[1:]))
            lis_t1 = [x+"_"+y for x, y in bigrams]
            feature['BiGram'] = lis_t1
            trigrams = list(zip(tokens[:-2], tokens[2:]))
            feature['TriGram'] = ["_".join(tri) for tri in trigrams]
        else:
            feature['Other'] = utt.text.strip("<>.,")
        features.append(feature)
    return pycrfsuite.ItemSequence(features)

write_file = sys.argv[2]

#Accuracy for baseline_crf.py
test(sys.argv[1], func_baseline(), write_file)

#Accuracy for advanced_crf.py
#test(sys.argv[1], func_advanced, write_file)