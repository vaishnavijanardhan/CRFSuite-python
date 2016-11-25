from hw3_corpus_tool import *
import os.path
import glob
import pycrfsuite
import sys


def convertToTag(dialog):
    x = [utt.act_tag for utt in dialog]
    return x

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
    print("accuracy:{} ".format(accuracy))
    return accuracy

def train(inputDIR, funcFeature, c1, c2, total_iterations):
    samples = get_data(inputDIR)
    trainer = pycrfsuite.Trainer(verbose=True)

    for index, dialog in enumerate(samples):
        features = funcFeature(dialog)
        tags = [utt.act_tag for utt in dialog]
        trainer.append(features, tags)

    trainer.set_params({
        'c1': c1,
        'c2': c2,
        'max_iterations': total_iterations,
        'feature.possible_transitions': True
    })

    trainer.train("model.crfsuite")

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
        output.write('Filename="{}"\n'.format(os.path.basename(dialog_filename)))
        for label in predLabel[index]:
            output.write(label + "\n")
        output.write("\n")
    output.close()

    if (not(None in trueLabel)):
        return calculateAccuracy(predLabel, trueLabel)

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

train(sys.argv[1], func_baseline, 1.0, 0.1, 100)
write_file = sys.argv[3]
test(sys.argv[2], func_baseline, write_file)