# Author   : Oguzhan Ozcelik
# Date     : 19.08.2022
# Subject  : SVM model for text classification
# Framework: scikit-learn

import os
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report


for lang in ['TR', 'EN']:
    print(f"Language: {'English' if lang=='EN' else 'Turkish'}")

    path = 'results/SVM/' + lang
    if not os.path.exists(path):
        os.makedirs(path)

    for fold in tqdm(range(5)):
        print(f"Fold: {fold}")
        train_pd = pd.read_csv(os.path.join('./dataset', lang, 'folds', lang+'_train_'+str(fold)+'.tsv'), sep='\t', encoding='utf-8')
        test_pd = pd.read_csv(os.path.join('./dataset', lang, 'folds', lang+'_test_'+str(fold)+'.tsv'), sep='\t', encoding='utf-8')

        train_pd = train_pd.dropna(axis=0)
        test_pd = test_pd.dropna(axis=0)

        all_texts = pd.concat([train_pd['text'], test_pd['text']]).values
        all_documents = [text for text in all_texts]

        train_x, test_x = [text for text in train_pd['text'].values], [text for text in test_pd['text'].values]
        train_y, test_y = [lbl for lbl in train_pd['label'].values], [lbl for lbl in test_pd['label'].values]

        tfidf_vector = TfidfVectorizer(max_features=5000)
        tfidf_vector.fit(all_documents)

        train_tfidf, test_tfidf = tfidf_vector.transform(train_x), tfidf_vector.transform(test_x)
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', verbose=True)
        SVM.fit(train_tfidf, train_y)
        pred_y = SVM.predict(test_tfidf)

        report = classification_report(test_y, pred_y, digits=4)
        print(report)

        with open(os.path.join(path, 'classification_report_'+str(fold)), 'w') as file:
            file.write(report)

