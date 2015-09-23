import data_utils.utils as du
import data_utils.ner as ner


# Load the starter word vectors
wv, word_to_num, num_to_word = ner.load_wv('data/ner/vocab.txt',
                                           'data/ner/wordVectors.txt')
tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
num_to_tag = dict(enumerate(tagnames))
tag_to_num = du.invert_dict(num_to_tag)

# Set window size
windowsize = 3

# Load the training set
docs = du.load_dataset('data/ner/train')
X_train, y_train = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                      wsize=windowsize)

# Load the dev set (for tuning hyperparameters)
docs = du.load_dataset('data/ner/dev')
X_dev, y_dev = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                  wsize=windowsize)

# Load the test set (dummy labels only)
docs = du.load_dataset('data/ner/test.masked')
X_test, y_test = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                    wsize=windowsize)

with open('inv_vocabulary_raw', 'w') as f1:
    with open('vocabulary_raw', 'w') as f2:
        lines1 = []
        lines2 = []
        for word, num in word_to_num.items():
            lines1.append(word + ' ' + str(num+1) + '\n')
            lines2.append(str(num+1) + ' ' + word + '\n')
        f1.writelines(lines1)
        f2.writelines(lines2)

with open('x_train', 'w') as f1:
    with open('y_train', 'w') as f2:
        lines1 = []
        lines2 = []
        for i in range(len(X_train)):
            lines1.append(' '.join([str(k+1) for k in X_train[i]]) + '\n')
            lines2.append(str(y_train[i]+1) + '\n')
        f1.writelines(lines1)
        f2.writelines(lines2)

with open('x_dev', 'w') as f1:
    with open('y_dev', 'w') as f2:
        lines1 = []
        lines2 = []
        for i in range(len(X_dev)):
            lines1.append(' '.join([str(k+1) for k in X_dev[i]]) + '\n')
            lines2.append(str(y_dev[i]+1) + '\n')
        f1.writelines(lines1)
        f2.writelines(lines2)

with open('word_vectors', 'w') as f:
    lines = []
    for i in range(len(wv)):
        lines.append(' '.join([str(k) for k in wv[i]]) + '\n')



