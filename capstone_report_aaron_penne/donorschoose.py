import os
import datetime

import numpy as np
random_state = 1138

import pandas as pd
pd.options.display.max_columns = 200
pd.options.mode.chained_assignment = None  # 'warn'

from wordcloud import WordCloud

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob

from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import model_selection

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


###############################################################################
# EXPLORE DATA
    
train_describe = train.describe(include='all')
    
# Compute correlation
encoded = cat_train.apply(lambda s: s.astype('category').cat.codes)
agg = pd.merge(encoded, num_train, how='left', left_index=True, right_index=True)
agg['labels'] = labels
corr = agg.corr()

# Find features that have moderate or above correlation
corr[corr['labels'].abs() > 0.03].index

# Correlation heatmap
fig, ax = plt.subplots(figsize=(7,7), dpi=200)
sns.heatmap(corr, ax=ax, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu', vmax=0.5, vmin=-0.5)
plt.title('Correlation Matrix of all features')
ax.figure.tight_layout()
ax.tick_params(axis='both', labelsize='xx-small')
plt.savefig('Correlation Matrix of all features.png', dpi=fig.dpi)


# Word clouds normalized
essay_accepted = txt_train.loc[labels==1, 'essay_agg_norm'].str.cat(sep=' ')
essay_rejected = txt_train.loc[labels==0, 'essay_agg_norm'].str.cat(sep=' ')
wc_accepted = WordCloud(width=800, height=400, 
                        max_words=75,
                        collocations=False,
                        background_color='white',
                        random_state=random_state).generate(essay_accepted)
plt.figure(figsize=(20,10))
plt.imshow(wc_accepted, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_accepted_norm.png', bbox_inches='tight')
wc_rejected = WordCloud(width=800, height=400, 
                        max_words=75,
                        collocations=False,
                        background_color='white',
                        random_state=random_state).generate(essay_rejected)
plt.figure(figsize=(20,10))
plt.imshow(wc_rejected, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_rejected_norm.png', bbox_inches='tight')

# Word clouds raw text
essay_accepted = txt_train.loc[labels==1, 'essay_agg'].str.cat(sep=' ')
essay_rejected = txt_train.loc[labels==0, 'essay_agg'].str.cat(sep=' ')
wc_accepted = WordCloud(width=800, height=400, 
                        max_words=75,
                        collocations=False,
                        background_color='white',
                        random_state=random_state).generate(essay_accepted)
plt.figure(figsize=(20,10))
plt.imshow(wc_accepted, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_accepted.png', bbox_inches='tight')
wc_rejected = WordCloud(width=800, height=400, 
                        max_words=75,
                        collocations=False,
                        background_color='white',
                        random_state=random_state).generate(essay_rejected)
plt.figure(figsize=(20,10))
plt.imshow(wc_rejected, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_rejected.png', bbox_inches='tight')

# Teacher prefix bar chart
teacher = pd.DataFrame(cat_train['teacher_prefix'])
teacher['labels'] = labels
sns.countplot(x='teacher_prefix', hue='labels', palette={0:'#1d686e', 1:'#a1def0'}, data=teacher)
plt.title('Rate of Acceptance per Teacher Prefix')
plt.savefig('Rate of Acceptance per Teacher Prefix.png', bbox_inches='tight')                   
                 
# Teacher prefix rates  
teacher_total = pd.DataFrame(teacher['teacher_prefix'].value_counts())
teacher_total['accepted'] = teacher.loc[teacher['labels']==1, 'teacher_prefix'].value_counts()
teacher_total['rejected'] = teacher.loc[teacher['labels']==0, 'teacher_prefix'].value_counts()
teacher_total['accepted_rate'] = teacher_total['accepted'] / teacher_total['teacher_prefix']
teacher_total['rejected_rate'] = teacher_total['rejected'] / teacher_total['teacher_prefix']
                                                         
###############################################################################
# REDUCE DATA
# Filter in/out features we care about
cols = ['res_descriptions_subjectivity', 
        'res_count', 
        'num_previous',
       'res_quantity', 
       'res_descriptions_word_count', 
       'essay_agg_word_count',
       'res_median_price', 
       'project_resource_summary_word_count',
       'essay_2_word_count', 
       'res_total_cost']
def select_columns(cat, num, cols):
    cat = cat[[c for c in cat if c in cols]]
    num = num[[c for c in num if c in cols]]
    return cat, num
cat_test, num_test = select_columns(cat_test, num_test, cols)
cat_train, num_train = select_columns(cat_train, num_train, cols)

txt_test = txt_test[[x for x in txt_test if '_norm' in x]]
txt_train = txt_train[[x for x in txt_train if '_norm' in x]]

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
# BALANCE CLASSES
def balance(df, labels):
    df_imbal = df.copy()
    df_imbal['labels'] = labels

    majority = df_imbal[df_imbal['labels']==1]
    minority = df_imbal[df_imbal['labels']==0]
    
    majority_dn = resample(majority,
                           n_samples=minority.shape[0],
                           replace=False,
                           random_state=random_state)
    minority_up = resample(minority,
                           n_samples=majority.shape[0],
                           replace=True,
                           random_state=random_state)
    
    df_bal_dn = pd.concat([minority, majority_dn])
    df_bal_up = pd.concat([minority_up, majority])
    
    return df_imbal, df_bal_dn, df_bal_up

num_train_imbal, num_train_dn, num_train_up = balance(num_train, labels)
txt_train_imbal, txt_train_dn, txt_train_up = balance(txt_train, labels)

# Pair plot
#def pair_heat(x, y, **kws):
#    cmap = sns.light_palette(kws.pop("color"), as_cmap=True)
#    plt.hist2d(x, y, cmap=cmap, cmin=1, **kws)
#g = sns.PairGrid(a, size=4)
#g.map_diag(plt.hist, bins=20)
#g.map_offdiag(pair_heat, bins=20)
#plt.savefig('pairplot_5000.png')
#
#g = sns.PairGrid(df_imbal, size=4, hue='labels', hue_order=[1,0], palette={0:'#1d686e', 1:'#a1def0'})
#g.map_diag(plt.hist, histtype='step', linewidth=3)
#g.map_offdiag(plt.scatter)
#g.add_legend()
#plt.savefig('pairplot_scatter_big2.png')

###############################################################################
# CROSS VALIDATION

all_scores = []
all_confusion = []

classifiers = [#('Dummy Classifier Baseline', DummyClassifier(random_state=random_state)),
#               ('Stochastic Gradient Descent', SGDClassifier(random_state=random_state)),
#               ('Logistic Regression', LogisticRegression(random_state=random_state)),
#               ('Linear SVM', LinearSVC(random_state=random_state)),
               ('Naive Bayes', GaussianNB())]

datasets = [('Imbalanced Dataset', num_train_imbal),
            ('Majority Downsampled', num_train_dn),
            ('Minority Upsampled', num_train_up)]

kfold = model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=random_state)
    
# Numerical/Categorical Data
for clf_name, clf in classifiers:
    all_scores = []
    all_confusion = []
    for df_name, df in datasets:
        
        x_train = df.copy()
        y_train = x_train['labels']
        x_train.drop(columns='labels', inplace=True)
        
        t = tic()
        scores = cross_val_score(clf, x_train, y_train, cv=kfold, scoring='roc_auc')
#        y_hat = cross_val_predict(clf, x_train, y_train, cv=kfold)
#        confusion = confusion_matrix(y_train, y_hat)
        print('{} with {} AUC: {:0.3f}  STD: {:0.3f}'.format(clf_name, df_name, np.median(scores), np.std(scores)))
        toc(t)
    
        all_scores.append(scores)
#        all_confusion.append(confusion)
        
    
    pd.DataFrame(all_scores).to_pickle('AUC ' + clf_name + '.pkl')
    pd.DataFrame(all_scores).to_pickle('Confusion ' + clf_name + '.pkl')
    
    fig, ax = plt.subplots(figsize=(6,4), dpi=200)
    sns.boxplot(data=all_scores, width=0.8, linewidth=0.8, ax=ax, color='#afafaf')
    sns.stripplot(data=all_scores, linewidth=0.8, ax=ax, color='black', alpha=0.3)
    plt.title(clf_name)
    plt.ylabel('AUC')
    plt.ylim(0.3, 0.8)
    ax.set_xticklabels([n for n,d in datasets])
    plt.savefig('AUC ' + clf_name + '.png', dpi=fig.dpi)



###############################################################################
# PREDICT
predictions = {}
prob = {}

#for col in txt_train:
#    clf = LinearSVC()
#    print(col)
#    print('Fitting transform...')
#    vct = TfidfVectorizer(ngram_range=(1,2), min_df=3)
#    x_train = vct.fit_transform(txt_train[col])
#
#    t = tic()
#    print('Fitting...')
#    clf.fit(x_train, labels)
#    x_test = vct.transform(txt_test[col])
#    print('Predicting...')
#    predictions[col] = clf.predict(x_test)
#    toc(t)

predictions = {}
prob = {}

#
## Random Forest on the categorical features
#print('## Random Forest ##')
#clf = RandomForestClassifier(max_depth=7, n_estimators=11, random_state=random_state)
#t = tic()
#print('Fitting...')
#clf.fit(num_train_up.drop, num_train_up['labels'])
#print('Predicting...')
#predictions['RF Num'] = clf.predict(num_test)
#prob['RF Num'] = clf.predict_proba(num_test)
#toc(t)


# SGD
# Numerical
print('## SGD ##')
x_train = num_train_up.drop(columns=['labels'])
y_train = num_train_up['labels']
x_test = num_test

clf = SGDClassifier(loss='log', random_state=random_state)
t = tic()
print('Fitting...')
clf.fit(x_train, y_train)
print('Predicting...')
predictions['sgd_num'] = clf.predict(x_test)
prob['sgd_num'] = clf.predict_proba(x_test)
toc(t)

# Text
col = 'essay_1_norm'
x_train = txt_train_up[col]
y_train = txt_train_up['labels']
x_test = txt_test[col]

clf = SGDClassifier(loss='log', random_state=random_state)
print('Fitting transform...')
vct = TfidfVectorizer(ngram_range=(1,3), min_df=3)
x_train = vct.fit_transform(x_train)
t = tic()
print('Fitting...')
clf.fit(x_train, y_train)
x_test = vct.transform(x_test)
print('Predicting...')
predictions[col] = clf.predict(x_test)
prob[col] = clf.predict_proba(x_test)
toc(t)

# Text
col = 'essay_2_norm'
x_train = txt_train_up[col]
y_train = txt_train_up['labels']
x_test = txt_test[col]

clf = SGDClassifier(loss='log', random_state=random_state)
print('Fitting transform...')
vct = TfidfVectorizer(ngram_range=(1,3), min_df=3)
x_train = vct.fit_transform(x_train)
t = tic()
print('Fitting...')
clf.fit(x_train, y_train)
x_test = vct.transform(x_test)
print('Predicting...')
predictions[col] = clf.predict(x_test)
prob[col] = clf.predict_proba(x_test)
toc(t)



predictions = {}
prob = {}


# Random Forest on the categorical features
print('## Random Forest ##')
clf = RandomForestClassifier(max_depth=7, n_estimators=11, random_state=random_state)
t = tic()
print('Fitting...')
clf.fit(num_train_up.drop(columns='labels'), num_train_up['labels'])
print('Predicting...')
predictions['RF Num'] = clf.predict(num_test)
prob['RF Num'] = clf.predict_proba(num_test)
toc(t)

## SVM ##

# Text
col = 'essay_1_norm'
x_train = txt_train_up[col]
y_train = txt_train_up['labels']
x_test = txt_test[col]

clf = CalibratedClassifierCV(LinearSVC(random_state=random_state))
print('Fitting transform...')
vct = TfidfVectorizer(ngram_range=(1,3), min_df=3)
x_train = vct.fit_transform(x_train)
scaler = preprocessing.StandardScaler(with_mean=False).fit(x_train)
t = tic()
print('Fitting...')
clf.fit(scaler.transform(x_train), y_train)
x_test = vct.transform(x_test)
print('Predicting...')
predictions[col] = clf.predict(scaler.transform(x_test))
prob[col] = clf.predict_proba(scaler.transform(x_test))
toc(t)

# Text
col = 'essay_2_norm'
x_train = txt_train_up[col]
y_train = txt_train_up['labels']
x_test = txt_test[col]

clf = CalibratedClassifierCV(LinearSVC(random_state=random_state))
print('Fitting transform...')
vct = TfidfVectorizer(ngram_range=(1,3), min_df=3)
x_train = vct.fit_transform(x_train)
scaler = preprocessing.StandardScaler(with_mean=False).fit(x_train)
t = tic()
print('Fitting...')
clf.fit(scaler.transform(x_train), y_train)
x_test = vct.transform(x_test)
print('Predicting...')
predictions[col] = clf.predict(scaler.transform(x_test))
prob[col] = clf.predict_proba(scaler.transform(x_test))
toc(t)


###############################################################################
# OUTPUT


# Vote by averaging probabilities
probabilities = [prob['essay_1_norm'], prob['essay_1_norm'], prob['essay_2_norm']]
avg_prob = np.mean(np.array(probabilities), axis=0)
prob_approve = [x[1] for x in avg_prob]
with open('prob_upsampled_essays.csv', 'w+') as f:
    f.write('{},{}\n'.format('id', 'project_is_approved'))
    for p_id, p in zip(test['id'], prob_approve):
        f.write('{},{}\n'.format(p_id, p))