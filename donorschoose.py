import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

pd.options.display.max_columns = 200

DEBUG = False

def tic():
    t = datetime.datetime.now()
    print()
    print(t)
    return t

def toc(t):
    t2 = datetime.datetime.now() - t
    print(t2)
    print()

def prep_data(df_applications, df_resources):
    print('Preprocessing data...')
    df = df_applications.copy()
    res = df_resources.copy()
    
    ## Process resources file first
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

    ## Application data
    df['teacher_number_of_previously_posted_projects'] = pd.to_numeric(df['teacher_number_of_previously_posted_projects'], errors='coerce')

    ## Combine resources with applications 
    df = pd.merge(df, res_unique[~res_unique.index.duplicated(keep='first')], how='left', on='id')

    ## Feature engineering
    # Split up subject categories into individual columns
    df['subject_a'], df['subject_b'] = df['project_subject_categories'].str.split(',', 1).str
    df['subject_c'], df['subject_d'] = df['project_subject_subcategories'].str.split(',', 1).str

    # Combine subject categories into single feature
    cols = ['subject_a', 'subject_b', 'subject_c', 'subject_d']
    for c in cols:
        df[c].fillna('', inplace=True)
        df[c] = df[c].apply(lambda s: s.strip())
    df['subject_agg'] = df[cols].apply(' '.join, axis=1)

    cols = ['project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']
    # Remove escaped characters from text
    for c in cols:
        df[c].fillna('', inplace=True)
        df[c] = df[c].apply(lambda s: s.replace('\\"', ' '))
        df[c] = df[c].apply(lambda s: s.replace('\\r', ' '))
        df[c] = df[c].apply(lambda s: s.replace('\\n', ' '))
        df[c] = df[c].apply(lambda s: s.strip())
        
    # Combine essays and project title into single feature
    df['essay_agg'] = df[['project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']].apply(' '.join, axis=1)

    # Split up datetime into columns
    df['date_year'] = df['project_submitted_datetime'].dt.year
    df['date_month'] = df['project_submitted_datetime'].dt.month
    df['date_dow'] = df['project_submitted_datetime'].dt.dayofweek
    df['date_doy'] = df['project_submitted_datetime'].dt.dayofyear
    df['date_hour'] = df['project_submitted_datetime'].dt.hour
    
    # Get word counts of each text feature
    cols = ['essay_agg', 'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'res_descriptions']
    for c in cols:    
        # Get word counts for each essay, etc.
        df['word_count_' + c] = df[c].apply(lambda s: len(s.split()))
    
    # Change column names for sanity
    df.rename({'teacher_number_of_previously_posted_projects': 'num_previous'}, inplace=True, axis='columns')
    
    # Set unique application ID as index
    df.set_index('id', inplace=True) 
    
    # Split out categorical features, text, and labels
    text = df[['project_title',
               'project_essay_1',
               'project_essay_2',
               'project_essay_3',
               'project_essay_4',
               'project_resource_summary',
               'essay_agg',
               'subject_agg',
               'res_descriptions'
             ]]
    categorical = df[['teacher_prefix',
                      'school_state',
                      'project_grade_category',
                      'subject_a',
                      'subject_b',
                      'subject_c',
                      'subject_d'
                    ]]
    numerical = df[['res_count', 
                    'res_quantity',
                    'res_median_price',
                    'res_total_cost',
                    'word_count_essay_agg', 
                    'word_count_project_title', 
                    'word_count_project_essay_1', 
                    'word_count_project_essay_2', 
                    'word_count_project_essay_3', 
                    'word_count_project_essay_4',
                    'num_previous'
                  ]]

    print('Converting categorical data to one-hot...')
    cat_one_hot = pd.get_dummies(categorical, drop_first=True)
    # Data to use for training/evaluating models
    df_model = pd.merge(cat_one_hot, numerical[~numerical.index.duplicated(keep='first')], how='left', left_index=True, right_index=True)
    df_text = text
    
    print('Normalizing text...')
    # Text normalization    
    stop_words = stopwords.words('english') + list(string.punctuation)
    stm = PorterStemmer()
    def normalize_text(s):
        tokens = [stm.stem(t.lower()) for t in word_tokenize(s) if t not in stop_words]
        normalized = ' '.join(tokens)
        return normalized
    df_text['essay_agg'] = df_text['essay_agg'].apply(normalize_text)

    print('Preprocessing data complete.')

    return df_model, df_text


# Set up directories
data_dir = 'data'
output_dir = 'output'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Read in all available data
print('Reading in data files...')
t = tic()
res = pd.read_csv(os.path.join(data_dir, 'resources.csv'), dtype='str')
train = pd.read_csv(os.path.join(data_dir, 'train.csv'), parse_dates=[4], dtype='str')
test = pd.read_csv(os.path.join(data_dir, 'test.csv'), parse_dates=[4], dtype='str')
toc(t)

if DEBUG:
    print('DEBUG = True')
    train = train.loc[:1000,:]
    test = test.loc[:1000,:]

labels = train['project_is_approved'].astype('int')
train.drop(columns=['project_is_approved'])

print('Preprocess train')
t = tic()
df_train, df_train_text = prep_data(train, res)
toc(t)

print('Preprocess test')
t = tic()
df_test, df_test_text = prep_data(test, res)
toc(t)

# Make the columns line up
cols = list(set(list(df_train.columns)) & set(list(df_test.columns)))
df_train = df_train[cols]
df_test = df_test[cols]



from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

print('* Random Forest with Categorical/Numerical Features')
clf = RandomForestClassifier(max_depth=7, n_estimators=11)

print('Cross validation...')
t = tic()
kfold = model_selection.KFold(n_splits=5, random_state=1138)
score = cross_val_score(clf, df_train, labels, cv=kfold, n_jobs=-1)
print('{} ACC: {:0.3f}%  STD: {:0.3f}%'.format('Random Forest', score.mean()*100.0, score.std()*100.0))
toc(t)

print('Fit and predict labels...')
t = tic()
# Predict based on categorical/numerical values
clf.fit(df_train, labels)
p_random_forest = clf.predict(df_test)
toc(t)



# http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/
## Support Vector Machine and TF-IDF
print('* Support Vector Machine on TF-IDF features')
clf = LinearSVC()

print('Fit transform...')
t = tic()
vct = TfidfVectorizer(ngram_range=(1,2), min_df=3)
x_train = vct.fit_transform(df_train_text['essay_agg'])
toc(t)

print('Cross validation...')
t = tic()
kfold = model_selection.KFold(n_splits=5, random_state=1138)
score = cross_val_score(clf, x_train, labels, cv=kfold, n_jobs=-1)
print('{} ACC: {:0.3f}%  STD: {:0.3f}%'.format('SVM TF-IDF', score.mean()*100.0, score.std()*100.0))
toc(t)

print('Fit and predict labels...')
t = tic()
clf.fit(x_train, labels)
x_test = vct.transform(df_test_text['essay_agg'])
p_svm = clf.predict(x_test)
toc(t)

# Vote by and-ing the predictions
predictions= []
for idx, val in enumerate(zip(p_svm, p_random_forest)):
    if val[0] & val[1]:
        predictions.append(1)
    else:
        predictions.append(0)
        
with open('predictions.csv', 'w+') as f:
    f.write('{},{}\n'.format('id', 'project_is_approved'))
    for p_id, p in zip(df_test.index, predictions):
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

