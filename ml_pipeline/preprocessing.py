# ml_pipeline/preprocessing.py
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(X_train, X_test, n_components):
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test
