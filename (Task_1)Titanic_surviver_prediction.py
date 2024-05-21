#Importing modules
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

#Reading the CSV file
df = pd.read_csv("Titanic-Dataset.csv")
df

df.columns

#Method-1: Using pipeline and ColumnTransformers
# Still need to deal with missing categories in test set
X = df.drop(columns=['Ticket', 'Fare', 'PassengerId', 'Name','Cabin',"Sex",'Embarked'])
y = df.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Creating Imputer for missing values of Age.
age_imputer = SimpleImputer(strategy='median')

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('age_imputer', age_imputer, ['Age'])
    ],
    remainder='passthrough'
)

# pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           #  ('minmax_scaler', minmax_scaler, ['range']),
                           ('lr', LinearRegression())
                           ]
                           )

# fit the pipeline
pipeline.fit(X_train, y_train)
#Score the Pipeline
pipeline.score(X_test, y_test)

# Method-2: Using Xgboost

# use xgboost to make a model and then get feature importances
import xgboost as xgb

# create X and y
X = df.drop(columns=['Ticket', 'Fare', 'PassengerId', 'Name','Cabin',"Sex",'Embarked'])
y = df.Survived

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X.assign(**X.select_dtypes(object).astype('category')),
    y, random_state=42)

xg = xgb.XGBRegressor(enable_categorical=True, random_state=42)
xg.fit(X_train, y_train)
xg.score(X_test, y_test)
