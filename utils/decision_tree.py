from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_decision_tree(x_train, y_train, x_test, y_test):
    # Parameter optimal manual
    optimal_params = {
    'criterion': 'gini',           # Menggunakan 'gini' untuk simplicity index
    'max_depth': 20,               # Memperbesar kedalaman pohon untuk lebih banyak pola
    'min_samples_split': 2,        # Membiarkan split minimal menjadi lebih fleksibel
    'min_samples_leaf': 1,         # Satu sampel minimum di daun
    'max_features': None,          # Gunakan semua fitur
    'class_weight': None           # Tidak menyesuaikan bobot kelas
}


    # Inisialisasi model Decision Tree dengan parameter optimal
    clf_dt = DecisionTreeClassifier(
        criterion=optimal_params['criterion'],
        max_depth=optimal_params['max_depth'],
        min_samples_split=optimal_params['min_samples_split'],
        min_samples_leaf=optimal_params['min_samples_leaf'],
        max_features=optimal_params['max_features'],
        class_weight=optimal_params['class_weight'],
        random_state=42
    )

    # Latih model pada data training
    clf_dt.fit(x_train, y_train)

    # Prediksi pada data testing
    predicted = clf_dt.predict(x_test)

    # Evaluasi performa
    accuracy = accuracy_score(y_test, predicted) * 100
    precision = precision_score(y_test, predicted, average='weighted') * 100
    recall = recall_score(y_test, predicted, average='weighted') * 100
    f1 = f1_score(y_test, predicted, average='weighted') * 100

    # Confusion Matrix dan Classification Report
    conf_matrix = confusion_matrix(y_test, predicted)
    class_report = classification_report(y_test, predicted)

    results = {
        "params_used": optimal_params,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
    }
    return results
