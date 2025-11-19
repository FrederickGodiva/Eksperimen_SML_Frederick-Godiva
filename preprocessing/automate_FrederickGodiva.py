import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load


def preprocess_data(data, target_column, save_path, file_path):
    data.drop(columns=['id', 'Unnamed: 32'], inplace=True)

    numeric_features = data.select_dtypes(
        include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(
        include=['object']).columns.tolist()
    column_names = list(data.columns)
    column_names.remove(target_column)

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

    features = data[column_names]
    target = data[target_column]

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.3, random_state=42)

    features_train = preprocessor.fit_transform(features_train)
    features_test = preprocessor.transform(features_test)

    dump(preprocessor, save_path)

    final_names = [col for col in column_names]
    final_names.extend([target_column])

    combined_train = np.column_stack((features_train, target_train))

    preprocess_data = pd.DataFrame(combined_train, columns=final_names)

    preprocess_data.to_csv(file_path, index=False)

    return features_train, features_test, target_train, target_test


def inference(data, preprocessor_path):
    pre = load(preprocessor_path)
    return pre.transform(data)


if __name__ == '__main__':
    data = pd.read_csv('breast_cancer_data.csv')
    target_column = 'diagnosis'
    save_path = 'preprocessing/preprocessor.joblib'
    file_path = 'preprocessing/processed_data.csv'

    X_train_t, X_test_t, y_train, y_test = preprocess_data(
        data=data,
        target_column=target_column,
        save_path=save_path,
        file_path=file_path
    )

    print("Preprocessing complete!")
