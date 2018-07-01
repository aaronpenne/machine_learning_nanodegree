import os
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import model_selection

pd.options.display.max_columns = 200
pd.options.mode.chained_assignment = None  # 'warn'

DEBUG = False

def tic():
    return datetime.datetime.now()

def toc(t):
    t2 = datetime.datetime.now() - t
    print(t2)
    print()

def prep_data(df_applications, df_resources):
    print('Preprocessing data...')
    df = df_applications.copy()
    res = df_resources.copy()


    ###########################################################################
    # PROCESS RESOURCES
    print('  Summarizing resources')
    # Convert numeric values to float
    res['quantity'] = pd.to_numeric(res['quantity'], errors='coerce')
    res['price'] = pd.to_numeric(res['price'], errors='coerce')
    # Calculate the total cost of each item
    res['cost'] = res['quantity'] * res['price']
    # Calculate the total cost of each proposal
    res_unique = res.groupby(by='id').agg({'description': 'count',
                                           'quantity': 'sum',
                                           'price': 'median',
                                           'cost': 'sum'})
    # Rename columns
    res_unique.columns = ['res_count', 'res_quantity', 'res_median_price', 'res_total_cost']
    # Replace all NaN descriptions with blanks
    res['description'].fillna('', inplace=True)
    # Combine the text of individual descriptions into one large text blob for each application
    res_unique['res_descriptions'] = res.groupby(by='id')['description'].apply(', '.join)


    ###########################################################################
    # PROCESS APPLICATIONS

    # Combine resources with applications
    df = pd.merge(df, res_unique[~res_unique.index.duplicated(keep='first')], how='left', on='id')

    # Convert numeric values to float
    df['teacher_number_of_previously_posted_projects'] = pd.to_numeric(df['teacher_number_of_previously_posted_projects'], errors='coerce')


    ###########################################################################
    # SPLIT DATA INTO TYPES

    # Split out categorical features, text, and labels
    cat = df[['teacher_prefix',
              'school_state',
              'project_grade_category',
              'project_subject_categories',
              'project_subject_subcategories',
              ]]

    num = df[['teacher_number_of_previously_posted_projects',
              'project_submitted_datetime',
              'res_count',
              'res_quantity',
              'res_median_price',
              'res_total_cost'
              ]]

    txt = df[['project_title',
              'project_essay_1',
              'project_essay_2',
              'project_essay_3',
              'project_essay_4',
              'project_resource_summary',
              'res_descriptions'
              ]]


    ###########################################################################
    # FEATURE ENGINEERING

    # Change column names for sanity
    num.rename({'teacher_number_of_previously_posted_projects': 'num_previous'}, inplace=True, axis='columns')

    # Fill any NaNs with blank
    txt.fillna('', inplace=True)

    # Remove escaped characters from text
    print('  Removing escape characters from text')
    txt = txt.applymap(lambda s: s.replace('\\"', ' '))
    txt = txt.applymap(lambda s: s.replace('\\r', ' '))
    txt = txt.applymap(lambda s: s.replace('\\n', ' '))
    txt = txt.applymap(lambda s: s.strip())

    # Split up subject categories into individual columns (max 2 subjects per category)
    print('  Splitting subjects')
    cat['subject_a'], cat['subject_b'] = cat['project_subject_categories'].str.split(',', 1).str
    cat['subject_c'], cat['subject_d'] = cat['project_subject_subcategories'].str.split(',', 1).str

    # Combine subject categories into single feature
    cols = ['subject_a', 'subject_b', 'subject_c', 'subject_d']
    for c in cols:
        cat[c].fillna('', inplace=True)
        cat[c] = cat[c].apply(lambda s: s.strip())
    cat['subject_agg'] = cat[cols].apply(' '.join, axis=1)

    # Deal with essay amount change from 4 to 2 by combining 1&2, 3&4 where appropriate
    print('  Combining essays')
    def new_essay_1(s):
        if s['project_essay_3'] == '':
            return s['project_essay_1']
        else:
            return s['project_essay_1'] + s['project_essay_2']
    txt['essay_1'] = txt.apply(new_essay_1, axis=1)

    def new_essay_2(s):
        if s['project_essay_3'] == '':
            return s['project_essay_2']
        else:
            return s['project_essay_3'] + s['project_essay_4']
    txt['essay_2'] = txt.apply(new_essay_2, axis=1)

    txt.drop(columns=['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', ], inplace=True)

    # Combine essays and project title into single feature
    txt['essay_agg'] = txt[['project_title', 'essay_1', 'essay_2']].apply(' '.join, axis=1)

    # Split up datetime into columns
    print('  Parsing dates')
    num['date_year'] = num['project_submitted_datetime'].dt.year
    num['date_month'] = num['project_submitted_datetime'].dt.month
    num['date_dow'] = num['project_submitted_datetime'].dt.dayofweek
    num['date_doy'] = num['project_submitted_datetime'].dt.dayofyear
    num['date_hour'] = num['project_submitted_datetime'].dt.hour
    num.drop(columns=['project_submitted_datetime'], inplace=True)

    # Get word count of every text feature
    print('  Counting words')
    word_count = txt.applymap(lambda s: len(s.split()))
    word_count = word_count.add_suffix('_word_count')
    num = pd.merge(num, word_count, left_index=True, right_index=True)

    # Get polarity and subjectivity of essays
#    print('  Getting polarity')
#    polarity = txt.applymap(lambda x: TextBlob(x).sentiment.polarity)
#    polarity = polarity.add_suffix('_polarity')
    print('  Getting subjectivity')
    subjectivity = txt.applymap(lambda x: TextBlob(x).sentiment.subjectivity)
    subjectivity = subjectivity.add_suffix('_subjectivity')
#    num = pd.merge(num, polarity, left_index=True, right_index=True)
    num = pd.merge(num, subjectivity, left_index=True, right_index=True)

    # Text normalization
    # FIXME Can this be sped up? Or done selectively?
    print('  Normalizing all text')
    stop_words = stopwords.words('english') + list(string.punctuation)
    stm = PorterStemmer()
    def normalize_text(s):
        tokens = [stm.stem(t.lower()) for t in word_tokenize(s) if t not in stop_words]
        normalized = ' '.join(tokens)
        return normalized
    txt_norm = txt.applymap(normalize_text)
    txt_norm = txt_norm.add_suffix('_norm')
    txt = pd.merge(txt, txt_norm, left_index=True, right_index=True)

    print('Preprocessing complete.')

    return cat, num, txt


###############################################################################
# SETUP

data_dir = 'data'
output_dir = 'output'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


###############################################################################
# READ IN DATA

print('## Read in Data ##')
t = tic()
try: res
except: res = pd.read_csv(os.path.join(data_dir, 'resources.csv'), dtype='str')

try: train
except: train = pd.read_csv(os.path.join(data_dir, 'train.csv'), parse_dates=[4], dtype='str')

try: test
except: test = pd.read_csv(os.path.join(data_dir, 'test.csv'), parse_dates=[4], dtype='str')
toc(t)

if DEBUG:
    print('DEBUG = True')
    train = train.head(10000)
    test = test.head(10000)


###############################################################################
# PREPROCESS DATA

print('## Preprocess Data ##')
labels = train['project_is_approved'].astype('int')
train.drop(columns=['project_is_approved'])

if not (os.path.isfile(os.path.join(output_dir, 'cat_train')) & os.path.isfile(os.path.join(output_dir, 'num_train')) & os.path.isfile(os.path.join(output_dir, 'txt_train'))):
    print('Training data')
    t = tic()
    cat_train, num_train, txt_train = prep_data(train, res)
    # Save files to pickles
    cat_train.to_pickle(os.path.join(output_dir, 'cat_train'))
    num_train.to_pickle(os.path.join(output_dir, 'num_train'))
    txt_train.to_pickle(os.path.join(output_dir, 'txt_train'))
    toc(t)
else:
    try: cat_train
    except: cat_train = pd.read_pickle(os.path.join(output_dir, 'cat_train'))

    try: num_train
    except: num_train = pd.read_pickle(os.path.join(output_dir, 'num_train'))

    try: txt_train
    except: txt_train = pd.read_pickle(os.path.join(output_dir, 'txt_train'))

if not (os.path.isfile(os.path.join(output_dir, 'cat_test')) & os.path.isfile(os.path.join(output_dir, 'num_test')) & os.path.isfile(os.path.join(output_dir, 'txt_test'))):
    print('Testing data')
    t = tic()
    cat_test, num_test, txt_test = prep_data(test, res)
    # Save files to pickles
    cat_test.to_pickle(os.path.join(output_dir, 'cat_test'))
    num_test.to_pickle(os.path.join(output_dir, 'num_test'))
    txt_test.to_pickle(os.path.join(output_dir, 'txt_test'))
    toc(t)
else:
    try: cat_test
    except: cat_test = pd.read_pickle(os.path.join(output_dir, 'cat_test'))

    try: num_test
    except: num_test = pd.read_pickle(os.path.join(output_dir, 'num_test'))

    try: txt_test
    except: txt_test = pd.read_pickle(os.path.join(output_dir, 'txt_test'))


# Compute correlation
agg = num_train.copy()
agg['labels'] = labels
corr = agg.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

# Filter in/out features we care about
cols = ['num_previous',
        'res_count',
        'res_median_price',
        'res_descriptions_word_count',
        'project_resource_summary_subjectivity',
        'essay_2_subjectivity',
        'essay_1_norm',
        'essay_2_norm',
        'essay_agg_norm',
        'res_descriptions_norm',
        'project_resource_summary_norm']
def select_columns(cat, num, txt, cols):
    cat = cat[[c for c in cat if c in cols]]
    num = num[[c for c in num if c in cols]]
    txt = txt[[c for c in txt if c in cols]]
    return cat, num, txt
cat_test, num_test, txt_test = select_columns(cat_test, num_test, txt_test, cols)
cat_train, num_train, txt_train = select_columns(cat_train, num_train, txt_train, cols)

# Convert categorical variables to one-hot
#cat_train_hot = pd.get_dummies(cat_train, drop_first=True)
#cat_test_hot = pd.get_dummies(cat_test, drop_first=True)

# Merge data into model ready dataframes
#num_train = pd.merge(num_train, cat_train_hot, left_index=True, right_index=True)
#num_test = pd.merge(num_test, cat_test_hot, left_index=True, right_index=True)

# Ensure that columns match in both train and test data
cols = list(set(list(num_train.columns)) & set(list(num_test.columns)))
num_train = num_train[cols]
num_test = num_test[cols]


###############################################################################
# CROSS VALIDATION
score = {}

print('## Cross validation ##')



kfold = model_selection.KFold(n_splits=5, random_state=1138)

t = tic()
clf = RandomForestClassifier(max_depth=15, n_estimators=11, random_state=1138)
score['RF Num F1'] = cross_val_score(clf, num_train, labels, cv=kfold, scoring='f1')
score['RF Num neg log loss'] = cross_val_score(clf, num_train, labels, cv=kfold, scoring='neg_log_loss')
score['RF Num ROC AUC'] = cross_val_score(clf, num_train, labels, cv=kfold, scoring='roc_auc')
y_hat = cross_val_predict(clf, num_train, labels, cv=kfold)
confusion = confusion_matrix(labels, y_hat)
tn, fp, fn, tp = confusion_matrix(labels, y_hat).ravel()
(tn, fp, fn, tp)
out = score['RF Num']
print('{} ACC: {:0.3f}%  STD: {:0.3f}%'.format('Random Forest', out.mean()*100.0, out.std()*100.0))
toc(t)

for col in txt_train:
    t = tic()
    clf = LinearSVC()
    vct = TfidfVectorizer(ngram_range=(1,2), min_df=3)
    x_train = vct.fit_transform(txt_train[col])
    score[col] = cross_val_score(clf, x_train, labels, cv=kfold, scoring='f1')
    out = score[col]
    print(col)
    print('{} ACC: {:0.3f}%  STD: {:0.3f}%'.format('SVM TF-IDF', out.mean()*100.0, out.std()*100.0))
    toc(t)


###############################################################################
# PREDICT
predictions = {}
prob = {}

# Random Forest on the categorical features
print('## Random Forest ##')
clf = RandomForestClassifier(max_depth=7, n_estimators=11)

t = tic()
print('Fitting...')
clf.fit(num_train, labels)
print('Predicting...')
predictions['RF Num'] = clf.predict(num_test)
prob['RF Num'] = clf.predict_proba(num_test)
toc(t)

# Support Vector Machine and TF-IDF
print('## Support Vector Machine ##')
for col in txt_train:
    clf = LinearSVC()
    print(col)
    print('Fitting transform...')
    vct = TfidfVectorizer(ngram_range=(1,2), min_df=3)
    x_train = vct.fit_transform(txt_train[col])

    t = tic()
    print('Fitting...')
    clf.fit(x_train, labels)
    x_test = vct.transform(txt_test[col])
    print('Predicting...')
    predictions[col] = clf.predict(x_test)
    toc(t)

# Probability testing
col = 'essay_1_norm'
clf = CalibratedClassifierCV(LinearSVC())
print(col)
print('Fitting transform...')
vct = TfidfVectorizer(ngram_range=(1,2), min_df=3)
x_train = vct.fit_transform(txt_train[col])
t = tic()
print('Fitting...')
clf.fit(x_train, labels)
x_test = vct.transform(txt_test[col])
print('Predicting...')
predictions[col] = clf.predict(x_test)
prob[col] = clf.predict_proba(x_test)
toc(t)

col = 'essay_2_norm'
clf = CalibratedClassifierCV(LinearSVC())
print(col)
print('Fitting transform...')
vct = TfidfVectorizer(ngram_range=(1,2), min_df=3)
x_train = vct.fit_transform(txt_train[col])
t = tic()
print('Fitting...')
clf.fit(x_train, labels)
x_test = vct.transform(txt_test[col])
print('Predicting...')
predictions[col] = clf.predict(x_test)
prob[col] = clf.predict_proba(x_test)
toc(t)


###############################################################################
# OUTPUT


# Vote by averaging probabilities
probabilities = [prob['RF Num'], prob['essay_1_norm'], prob['essay_2_norm']]
avg_prob = np.mean(np.array(probabilities), axis=0)
prob_approve = [x[1] for x in avg_prob]
with open('prob.csv', 'w+') as f:
    f.write('{},{}\n'.format('id', 'project_is_approved'))
    for p_id, p in zip(test['id'], prob_approve):
        f.write('{},{}\n'.format(p_id, p))






# ## Categorical prediction (cross validation testing)
#
#from sklearn import model_selection
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.naive_bayes import GaussianNB
#
#names = [#'Nearest Neighbors',
#         #'Gaussian Process',
#         'Decision Tree',
#         'Random Forest',
#         'Neural Net',
#         #'Naive Bayes'
#        ]
#classifiers = [#KNeighborsClassifier(3),
#               #GaussianProcessClassifier(1.0 * RBF(1.0)),
#               DecisionTreeClassifier(max_depth=5),
#               RandomForestClassifier(max_depth=5, n_estimators=10),
#               MLPClassifier(alpha=1),
#               #GaussianNB()
#              ]
#
#
##cat_corr = categorical[['school_state',
##                        'project_grade_category',
##                        'subject_b']]
##cat_one_hot = pd.get_dummies(cat_corr, drop_first=True)
#rem_corr = remaining[['res_count',
#                      'res_quantity',
#                      'res_median_price',
#                      'word_count_essay_agg',
#                      'word_count_project_essay_2',
#                      'num_previous']]
## Data to use for training/evaluating models
##df_model_corr = pd.merge(cat_one_hot, rem_corr[~rem_corr.index.duplicated(keep='first')], how='left', left_index=True, right_index=True)
#df_model_corr = rem_corr.copy()
#
#if TRAIN:
#    labels_series = labels['project_is_approved']
#
#    kfold = model_selection.KFold(n_splits=5, random_state=1138)
#
#    for name, clf in zip(names, classifiers):
#        out = model_selection.cross_val_score(clf, df_model, labels_series, cv=kfold)
#        print('{} ACC: {:0.3f}%  STD: {:0.3f}%'.format(name, out.mean()*100.0, out.std()*100.0))
    #
#Results from various predictive models on categorical data only (first pass, no category split):
#    Decision Tree ACC: 84.755%  STD: 0.128%
#    Random Forest ACC: 84.768%  STD: 0.131%
#    Neural Net ACC: 84.027%  STD: 0.846%
#
#Results from preprocessed data (with dates):
#    Decision Tree ACC: 84.765%  STD: 0.129%
#    Random Forest ACC: 84.768%  STD: 0.131%
#    Neural Net ACC: 79.964%  STD: 8.971%
#
#Results from preprocessed data (without dates):
#    Decision Tree ACC: 84.765%  STD: 0.129%
#    Random Forest ACC: 84.768%  STD: 0.131%
#    Neural Net ACC: 84.249%  STD: 0.823%
#
#Results from preprocessed data (without subjects or dates):
#    Decision Tree ACC: 84.767%  STD: 0.130%
#    Random Forest ACC: 84.768%  STD: 0.131%
#    Neural Net ACC: 82.698%  STD: 3.170%
#
#Results from only columns that had correlation > ||0.15||
#['school_state', 'project_grade_category', 'res_quantity', 'word_count_project_title', 'date_year']
#    Decision Tree ACC: 87.273%  STD: 8.509%
#    Random Forest ACC: 84.182%  STD: 6.545%
#    Neural Net ACC: 84.182%  STD: 6.545%


## Attempt to do basic text classification

#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.svm import LinearSVC
#
#kfold = model_selection.KFold(n_splits=5, random_state=1138)
#vct = TfidfVectorizer()
#vct_train = vct.fit_transform(text['essay_norm'])
#clf = LinearSVC()
#
#out = model_selection.cross_val_score(clf, vct_train, labels, cv=kfold)
#print('ACC: {:0.3f}%  STD: {:0.3f}%'.format(out.mean()*100.0, out.std()*100.0))Results for aggregated essays (with escaped chars removed, but not normalized at all):
##    ACC: 84.822%  STD: 0.133%
##
##Results for aggregated essays (with escaped chars removed, but not normalized at all):
##    ACC: 84.854%  STD: 0.119%
#
## ## Predict output with test data #FIXME Functionalize everything so it can be done on train and test
#clf = RandomForestClassifier(max_depth=5, n_estimators=10)
#clf.fit(rf_train, labels)
#out_rf = clf.predict(rf_test)vct = TfidfVectorizer()
#vct_train = vct.fit_transform(text['essay_norm'])
#vct_test = vct.transform(test['essay_norm'])
#
#clf = LinearSVC()
#clf.fit(vct_train, labels)
#out_nlp = clf.predict(vct_test)




#
#
# ####  Fancy embedding approach with sentence2vec
#
## https://www.kaggle.com/shivamb/extensive-text-data-feature-engineering/notebook
## http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/
## https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
##
#
#xtrain = text['essay_norm'].values
#
#
## https://fasttext.cc/docs/en/english-vectors.html
#import io
#
#def load_vectors(fname):
#    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#    n, d = map(int, fin.readline().split())
#    data = {}
#    for line in fin:
#        tokens = line.rstrip().split(' ')
#        data[tokens[0]] = map(float, tokens[1:])
#    return data
#
##f = open(EMBEDDING_FILE)
##for line in f:
##    if run_for_small_data and len(embeddings_index) == 100:
##      break
##    values = line.split()
##    word = values[0]
##    coefs = np.asarray(values[1:], dtype='float32')
##    embeddings_index[word] = coefs
##f.close()
#
#from nltk import word_tokenize
## FIXME file name
#embedding_vector_pretrained = load_vectors('FILENAMEHERE')
#
#def sent2vec(sent):
#    matrix = []
#    tokens = word_tokenize(sent)
#    for word in tokens:
#        if word in embedding_vector_pretrained:
#            matrix.append(embedding_vector_pretrained[word])
#        if not word.isalpha():
#            continue
#    matrix = np.array(matrix)
#    vector = matrix.sum(axis=0)
#    if type(vector) != np.ndarray:
#        return np.zeros(300)
#    return vector / np.sqrt((vector ** 2).sum())
#
#xtrain_vector = np.array([sent2vec(sent) for sent in xtrain])
#xtrain_vector[10]
