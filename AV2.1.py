import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Carregando os dados
# Substitua o caminho pelo local do seu dataset
df = pd.read_csv("Skyserver.csv")

# Features
X = df.drop(columns=['objid', 'specobjid', 'class'])

# Target
y = df['class']

# Colunas categóricas
categorical_columns = ['run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid']

# Colunas numéricas
numerical_columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']

# 2. Pré-processamento
# Definindo o pré-processamento das colunas numéricas e categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combinando transformações numéricas e categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


