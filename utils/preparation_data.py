# utils.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def process_tfidf(df, text_column='stemming', label_column='label_encoded'):
    """
    Function to apply TF-IDF on the specified text column.
    Returns the TF-IDF transformed feature matrix and the corresponding labels.
    """
    # Ensure that the text is of type string
    tvec = TfidfVectorizer()
    feature_tf = tvec.fit_transform(df[text_column].astype('U'))
    text_tf = pd.DataFrame(feature_tf.todense(), columns=tvec.get_feature_names_out())

    # Get features and labels
    X = text_tf
    y = df[label_column]
    
    return X, y, tvec

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Function to split the data into training and testing sets.
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return x_train, x_test, y_train, y_test
