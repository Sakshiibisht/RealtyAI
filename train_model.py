import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# ---------------- LOAD DATA ----------------

df = pd.read_csv("data/housing.csv")

print("Dataset Loaded")
print(df.head())

# ---------------- SELECT IMPORTANT COLUMNS ----------------

df = df[["location","size","total_sqft","bath","balcony","price"]]

# ---------------- CLEAN DATA ----------------

df = df.dropna()

# Convert sqft to numeric
df["total_sqft"] = pd.to_numeric(df["total_sqft"], errors="coerce")

# Extract bedrooms from size column
df["bedrooms"] = df["size"].str.extract("(\d+)")

df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")

df = df.dropna()

# ---------------- ENCODE LOCATION ----------------

le = LabelEncoder()

df["location"] = le.fit_transform(df["location"])

# ---------------- FEATURES AND TARGET ----------------

X = df[["location","total_sqft","bath","balcony","bedrooms"]]

y = df["price"]

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------

model = RandomForestRegressor(n_estimators=200)

model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------

pred = model.predict(X_test)

score = r2_score(y_test, pred)

print("Model Accuracy:", score)

# ---------------- SAVE MODEL ----------------

pickle.dump(model, open("price_model.pkl", "wb"))


print("Model Saved Successfully")
