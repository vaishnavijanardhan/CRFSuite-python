from hw3_corpus_tool import *
import os.path
import glob
import pycrfsuite
import sys

def func_advanced(dialog):
    features = []
    for index, utt in enumerate(dialog):
        feature = {}
        if index == 0:
            feature["FirstUtt"] = 1
        if index > 0 and not(dialog[index].speaker == dialog[index-1].speaker):
            feature["Speaker_Changed"] = 1
        if (utt.pos):
            tokens = [word.token for word in utt.pos]
            feature['Token'] = tokens
            lis_t = [word.pos for word in utt.pos]
            feature['PartOfSpeech'] = lis_t
            leng_t = len(utt.pos)
            feature['Length'] = leng_t
            feature['START_WITH'] = utt.pos[0].token
            bigrams = list(zip(tokens[:-1], tokens[1:]))
            lis_t1 = [x+"_"+y for x, y in bigrams]
            feature['BiGram'] = lis_t1
            if(utt.pos[-1].token == '?'):
                feature['Statement'] = 'Question'
            else:
                feature['Statement'] = 'Answer'
            trigrams = list(zip(tokens[:-2], tokens[2:]))
            feature['TriGram'] = ["_".join(tri) for tri in trigrams]
        else:
            feature['Other'] = utt.text.strip("<>.,")
        features.append(feature)
    return pycrfsuite.ItemSequence(features)


def train(train_dir, feature_ext_fn, c1, c2, total_iterations):
    samples = get_data(train_dir)
    trainer = pycrfsuite.Trainer(verbose=True)

    for index, dialog in enumerate(samples):
        features = feature_ext_fn(dialog)
        tags = [utt.act_tag for utt in dialog]
        trainer.append(features, tags)
    i = 0
    trainer.set_params({
        'c1': c1,
        'c2': c2,
        'max_iterations': total_iterations,
        'feature.possible_transitions': True
    })
    trainer.train("model.crfsuite")

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

def convertToTag(dialog):
    x = [utt.act_tag for utt in dialog]
    return x

def test(test_dir, feature_ext_fn, write_file):
    trueLabel = []
    predLabel = []
    output = open(write_file, 'w')
    tagger = pycrfsuite.Tagger()
    tagger.open("model.crfsuite")
    dialog_filenames = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

    for dialog_filename in dialog_filenames:
        dialog = get_utterances_from_filename(dialog_filename)
        xseq = feature_ext_fn(dialog)
        predLabel.append(tagger.tag(xseq))
        trueLabel.append(convertToTag(dialog))

    for index, dialog_filename in enumerate(dialog_filenames):
        output.write('Filename="{}"\n'.format(os.path.basename(dialog_filename)))
        for label in predLabel[index]:
            output.write("{}\n".format(label))
        output.write("\n")
    output.close()

    if (not(None in trueLabel)):
        return calculateAccuracy(predLabel, trueLabel)

train(sys.argv[1], func_advanced, 3.0, 0.1, 100)
write_file=sys.argv[3]
test(sys.argv[2], func_advanced, write_file)