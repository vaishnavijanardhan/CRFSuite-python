from hw3_corpus_tool import *
import os.path
import glob
import pycrfsuite
import sys

def dlg2tags(dialog):
    x = [utt.act_tag for utt in dialog]
    return x

def basic_dlg2feat(dialog):
    features = []
    for idx, utt in enumerate(dialog):
        feature = []
        if utt.pos:
            feature = feature + ["POS_" + word.pos for word in utt.pos]
            feature = feature + ["TOKEN_" + word.token for word in utt.pos]
        if (idx == 0):
            feature.append("FEAT_begin")
        if (idx > 0 and dialog[idx].speaker != dialog[idx-1].speaker):
            feature.append("FEAT_change_speaker")
        features.append(feature)
    return features

def compute_accuracy(ypred, ytrue):
    acc = 0
    count = 0
    for i in range(len(ypred)):
        for j in range(len(ypred[i])):
            count = count + 1
            if ypred[i][j] == ytrue[i][j]:
                acc = acc + 1
    acc = float(acc) / float(count)
    print("accuracy: %f" % acc)
    return acc

def train(train_dir, feature_ext_fn, c1=1.0, c2=0.1, nitr=100):
    print("processing data...")
    samples = get_data(train_dir)
    trainer = pycrfsuite.Trainer(verbose=True)
    for idx, dialog in enumerate(samples):
        if idx+1 % 100 == 0:
            print("%d dialogs processed" % idx+1)
        features = feature_ext_fn(dialog)
        tags = dlg2tags(dialog)
        trainer.append(features, tags)
    trainer.set_params({
        'c1': c1,   # coefficient for L1 penalty
        'c2': c2,  # coefficient for L2 penalty
        'max_iterations': nitr,  # stop earlier
        'feature.possible_transitions': True
    })
    print("start training")
    trainer.train("model.crfsuite")
    print("training finished, model saved to model.crfsuite")

def test(test_dir, feature_ext_fn, write_file):
    print("start testing...")
    tagger = pycrfsuite.Tagger()
    tagger.open("model.crfsuite")
    dialog_filenames = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    ytrue, ypred = [], []

    for dialog_filename in dialog_filenames:
        dialog = get_utterances_from_filename(dialog_filename)
        xseq = feature_ext_fn(dialog)
        ypred.append(tagger.tag(xseq))
        ytrue.append(dlg2tags(dialog))

    output = open(write_file, 'w')
    for idx, dialog_filename in enumerate(dialog_filenames):
        output.write('Filename="%s"\n' % os.path.basename(dialog_filename))
        for label in ypred[idx]:
            output.write("%s\n" % label)
        output.write("\n")
    output.close()
    print("testing finished, results saved in %s." % write_file)

    if None not in ytrue:
        print("true labels found, calculating accuracy...")
        return compute_accuracy(ypred, ytrue)

train(sys.argv[1], basic_dlg2feat, 1.0, 0.1, 100)
write_file = sys.argv[3]
test(sys.argv[2], basic_dlg2feat, write_file)
