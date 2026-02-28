import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and CatBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

print("="*80)
print("          ADHD MODEL ACCURACY BOOST: +93% TARGET")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'adhd_ratio_70.csv')
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()

# ============================================================================
# STEP 2: PRODUCTION-FRIENDLY FEATURE ENGINEERING
# ============================================================================
def get_features(df):
    df_new = df.copy()
    
    # Basic Symptom metrics
    df_new['SymptomSum'] = df_new['InattentionScore'] + df_new['HyperactivityScore'] + df_new['ImpulsivityScore']
    df_new['Inatt_Hyper_Inter'] = df_new['InattentionScore'] * df_new['HyperactivityScore']
    
    # Lifestyle impact
    st_col = 'ScreenTimeHours' if 'ScreenTimeHours' in df_new.columns else 'ScreenTime'
    df_new['Screen_Sleep_Ratio'] = df_new[st_col] / (df_new['SleepHours'] + 0.1)
    
    # Critical Ratios
    df_new['Symptom_Age_Ratio'] = df_new['SymptomSum'] / (df_new['Age'] + 1)
    
    return df_new

data_engineered = get_features(data)

# ============================================================================
# STEP 3: PREPROCESSING
# ============================================================================
target = 'ADHD'
X = data_engineered.drop(target, axis=1)
y = data_engineered[target]

for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna('None')

label_encoders = {}
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# ============================================================================
# STEP 4: BALANCED SPLIT & ENSEMBLE
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Upsample
train_data = pd.concat([X_train, y_train], axis=1)
maj = train_data[train_data[target] == 1]
min = train_data[train_data[target] == 0]
min_up = resample(min, replace=True, n_samples=len(maj), random_state=42)
train_bal = pd.concat([maj, min_up]).sample(frac=1, random_state=42)
X_train_bal = train_bal.drop(target, axis=1)
y_train_bal = train_bal[target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Ensemble
rf = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
gb = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_depth=7, random_state=42)

estimators = [('rf', rf), ('gb', gb)]
if HAS_XGB: estimators.append(('xgb', XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=8, colsample_bytree=0.7, random_state=42)))
if HAS_CATBOOST: estimators.append(('cat', CatBoostClassifier(iterations=1000, learning_rate=0.01, depth=8, random_state=42, verbose=False)))

stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
stacking.fit(X_train_scaled, y_train_bal)

acc = accuracy_score(y_test, stacking.predict(X_test_scaled))
print(f"\n[OK] Final Accuracy: {acc*100:.2f}%")

# ============================================================================
# STEP 8: SAVE
# ============================================================================
joblib.dump(stacking, 'adhdModel.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'labelEncoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("\nDONE! Models updated.")
