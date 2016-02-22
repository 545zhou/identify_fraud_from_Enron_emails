#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### To select which features I should choose, I used two methods. 
### One method is learned from previous project. I'll draw the histogram of each feature, with the entried being divided to "poi" and "nopoi". 
### By looking into the histograms, I can see if there is a pattern and decide if that features should be chosen to do the prediction.
### Another method is to use the sklearn.feature_selection module as instructed in the class. 
### By combining the result from these two method, I can hava an idea which features are important.
### I at first select all the possible features in the beginning and then plot histgrams of each feature
### to investigate the correlation between features and "poi". Then I choose a few important features.
### The feature "email_address" is a string not a number, so I remove it in the first place.

features_list_without_poi = ['salary', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
features_list = features_list_without_poi[:]
features_list.insert(0, 'poi')
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### At the first glance of the data, I found this is a person called "TOTAL". It's obivious that this entry should be removed.
data_dict.pop("TOTAL")

feature_value_list_poi = []
feature_value_list_nopoi = []
for i in range(0, 19):
    feature_value_list_poi.append([])
    feature_value_list_nopoi.append([])
    
features_with_NaN = {'salary': 0, 'deferral_payments': 0, 'total_payments': 0, 
                 'loan_advances': 0, 'bonus': 0, 'restricted_stock_deferred': 0, 
                 'deferred_income': 0, 'total_stock_value': 0, 'expenses': 0, 
                 'exercised_stock_options': 0, 'other': 0, 'long_term_incentive': 0,
                 'restricted_stock': 0, 'director_fees': 0,'to_messages': 0, 'from_poi_to_this_person': 0, 'from_messages': 0,
                 'from_this_person_to_poi': 0, 'shared_receipt_with_poi': 0}

for person in data_dict.values():
    for i in range(0, 19):
        if person['poi'] == True:
            if person[features_list_without_poi[i]] != 'NaN':
                feature_value_list_poi[i].append(person[features_list_without_poi[i]])
            else:
                feature_value_list_poi[i].append(0);
                features_with_NaN[features_list_without_poi[i]] += 1
        else:
            if person[features_list_without_poi[i]] != 'NaN':
                feature_value_list_nopoi[i].append(person[features_list_without_poi[i]])
            else:
                feature_value_list_nopoi[i].append(0);
                features_with_NaN[features_list_without_poi[i]] += 1
    
import pylab as pl
pl.close('all')

### Now draw the histogram of each feature with people being diveded into two groups "poi" and "no poi". 
### I found that for features "loan_advances", "restricted_stock_deferred" and "director_fees", 
### there is none or only one entry in "poi" group. So these are not important features. I will skip them.
k = 0
for i in range(0, 19):
    if( features_list_without_poi[i] == "loan_advances" or features_list_without_poi[i] 
    == "restricted_stock_deferred" or features_list_without_poi[i] == "director_fees"):
        continue

    pl.figure(k / 4 + 1)
    pl.subplot(2, 2, k % 4 + 1)  
    k = k + 1
    pl.hist(feature_value_list_poi[i], bins = 50, histtype='bar', normed=True, color='r', label='poi')
    pl.hist(feature_value_list_nopoi[i], bins = 50, histtype='bar', normed=True, alpha = 0.5, color='b', label='no poi')
    pl.legend()
    pl.title(features_list_without_poi[i])
    
### From the generated plots, I choose 6 subplots in which I think we can see the distributions of "poi" and "no poi" are different.
### These 6 subplots are corresponding to 6 features: 'bonus', 'salary', 'expenses', 'to_messages', 'shared_receipt_with_poi', and 'exercised_stock_options'.
### But the features are not decided yet. Let's check the second method using the sklearn.feature_selection module.

data = featureFormat(data_dict, features_list, remove_all_zeroes=False, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = 0.3, random_state = 42)

###I want to order the featurs according to their importance.
from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile = 100)
selector.fit(features_train, labels_train)
important_features_with_score = zip(features_list_without_poi, selector.scores_)
important_features_with_score.sort(key = lambda x: x[1], reverse = True)

## Print the features according to their importance.
for item in important_features_with_score:
    print item

### From the above printed result, we see 6 most important features are "'bonus','exercised_stock_options',
### 'total_stock_value', 'salary','total_payments', 'shared_receipt_with_poi'. 

### Using method 2 to select features are more objective. We now know the important features. But this selection process is not over yet. 
### Later I'm going to remove outliers and recalculate the importance of features and then decide which features I will select.

### Task 2: Remove outliers
### I would draw 5 histograms corresponding 5 features to check the outlier.
        
### By looking at feature "total_stock_value", I found there is one pearson called "BELFER ROBERT" having negative total_stock_value. This entry should be deleted.

data_dict.pop("BELFER ROBERT")

### After removing the outliers, lets check the importance of features again
data = featureFormat(data_dict, features_list, remove_all_zeroes=False, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = 0.3, random_state = 42)
selector = SelectPercentile(f_classif, percentile = 100)
selector.fit(features_train, labels_train)
important_features_with_score = zip(features_list_without_poi, selector.scores_)
important_features_with_score.sort(key = lambda x: x[1], reverse = True)

## Print the features according to their importance.
print "After removing outliers, let us check the importance of features calculated by SelectedPercntile again"
for item in important_features_with_score:
    print item
features_sorted_by_score = []
for i in range(0, 19):
    features_sorted_by_score.append(important_features_with_score[i][0])
    
### Now use decision tree to calculate the importance
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(features, labels)
feature_importances_by_DT = zip(features_list_without_poi, clf.feature_importances_)
feature_importances_by_DT.sort(key = lambda x: x[1], reverse = True)
print "Check the importance of features calculated by Decision Tree"
for item in feature_importances_by_DT:
    print item

features_sorted_by_score_by_DT = []
for i in range(0, 19):
    features_sorted_by_score_by_DT.append(feature_importances_by_DT[i][0])  



### Check how many features I should select

### For GaussianNB
#uncomment to run
#from sklearn.naive_bayes import GaussianNB
#for i in range(1, 19):
#    important_features = features_sorted_by_score[0:i]
#    important_features.insert(0, 'poi')
#    
#    clf = GaussianNB()
#    print i
#    test_classifier(clf, data_dict, important_features)

### For Decision Tree
#uncomment to run
#for i in range(1, 19):
#    important_features = features_sorted_by_score_by_DT[0:i]
#    important_features.insert(0, 'poi')
#    
#    clf = tree.DecisionTreeClassifier()
#    print i
#    test_classifier(clf, data_dict, important_features)

important_features = ['bonus', 'salary','shared_receipt_with_poi','deferred_income','exercised_stock_options','total_stock_value', 'expenses']
important_features.insert(0, 'poi')

important_features_by_DT = ['exercised_stock_options', 'bonus']
important_features_by_DT.insert(0, 'poi')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

### In the lecture, instructor was using the ratio of 'from_poi_to_this_person' / 'to_messages' 
### and ratio of 'from_this_perston_to_poi' / 'from_messages' to show how to create new features. I want to try some thing new. 
### The feature I created is called 'long_term_centive_ratio'. Its defination is 'long_term_centive' divided by 'total_payments'. 

my_dataset = data_dict
for person in my_dataset.values():
    if (person['total_payments'] != 'NaN'
    and person['long_term_incentive'] != 'NaN'
    and person['total_payments'] != 0):
        person['long_term_incentive_ratio'] = float(person['long_term_incentive']) / person['total_payments']
    else:
        person['long_term_incentive_ratio'] = 0;
### What I think is that for 'poi's, the 'long_term_centive_ratio' may be higher than 'nopoi's. 
### To verify my thought, let me draw a histogram using the new created feature and also check if adding this feature increase the final preision or recall.
ratio_value_list_poi = []
ratio_value_list_nopoi = []
    
for person in my_dataset.values():
    if person['poi'] == True:
        ratio_value_list_poi.append(person['long_term_incentive_ratio'])
    else:
        ratio_value_list_nopoi.append(person['long_term_incentive_ratio'])

pl.figure(5)
pl.hist(ratio_value_list_poi, bins = 200, histtype='bar', normed=True, color='r', label='poi')
pl.hist(ratio_value_list_nopoi, bins = 50, histtype='bar', normed=True, alpha = 0.5, color='b', label='no poi')
pl.legend()
pl.title('long_term_incentive_ratio')
### From the histogram, it seems 'poi's are more likely to have higher "long_term_incentive_ratio". 
### Moreover, when I use this ratio in the given GaussianNB() classifier to test prediction, I found the performance with this new feature is better. So I will use it.
important_features.append('long_term_incentive_ratio')
#important_features_by_DT.append('long_term_incentive_ratio')
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### The classifiers I will try are Naivie Bayes, SVM and Decision Tree.

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import tree
from sklearn.decomposition import PCA
print "Testing Naive Bayes classifier"
### Naive Bayes method doesn't require rescaling the features.
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
test_classifier(clf, my_dataset, important_features)

print "Testing SVM classifier"
#clf = SVC()
#test_classifier(clf, my_dataset, important_features)
### There is an error when runing SVM, so I will skip it.
print "SVM classifier does not work on the dataset, so skip it"

print "Testing Decision Tree classifier"
### Decision also does not need rescaling the features.
clf = tree.DecisionTreeClassifier(random_state=42)
test_classifier(clf, my_dataset, important_features_by_DT)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### I decide to use Decision Tree to be the classifier to make my final decision. Although the accuracy of GaussianNB() is a little higher than DecisionTreeClassifier(),
### the DecisionTreeClassifier() gives me more balanced precision and recall values. DecisionTreeClassifier() also has the potential to improve by tuning the parameters.

### uncomment to run, it will take a few minutes
#for pca_n_component in [None, 1, 2]:
#    for dt_min_samples_split in [2, 4, 6, 8]:
#        for dt_max_depth in [None, 5, 10, 15]:
#            print [pca_n_component, dt_min_samples_split, dt_max_depth]
#            clf = Pipeline([('pca', PCA(n_components = pca_n_component)), ('dt', tree.DecisionTreeClassifier(min_samples_split = dt_min_samples_split,max_depth = dt_max_depth, random_state= 42))])
#            test_classifier(clf, my_dataset, important_features_by_DT)
            
features_list = important_features_by_DT
clf = Pipeline([('pca', PCA(n_components = 2)), ('dt', tree.DecisionTreeClassifier(min_samples_split = 8,max_depth = 5, random_state= 42))])
test_classifier(clf, my_dataset, features_list)
### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)