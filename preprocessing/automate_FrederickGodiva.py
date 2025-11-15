import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(data, target_column, save_path, file_path):
    numeric_features = data.select_dtypes(
        include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(
        include=['object']).columns.tolist()
    column_names = data.columns
    column_names = data.columns.drop(target_column)

    df_header = pd.DataFrame(columns=column_names)

    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    features = data.drop(columns=[target_column])
    target = data[target_column]

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.3, random_state=42)

    features_train = preprocessor.fit_transform(features_train)
    features_test = preprocessor.transform(features_test)

    dump(preprocessor, save_path)

    return features_train, features_test, target_train, target_test
