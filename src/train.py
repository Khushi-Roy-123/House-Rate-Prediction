import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Set random seed
np.random.seed(42)

def load_data():
    print("Loading data...")
    # Assume running from project root
    if os.path.exists('data/train.csv'):
        train_path = 'data/train.csv'
        test_path = 'data/test.csv'
    elif os.path.exists('../data/train.csv'):
         # Fallback if running from src/
        train_path = '../data/train.csv'
        test_path = '../data/test.csv'
    else:
        raise FileNotFoundError("Data not found in data/ or ../data/")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def feature_engineering(train_df, test_df):
    print("Performing Feature Engineering...")
    
    # Save Ids
    train_ids = train_df['Id']
    test_ids = test_df['Id']
    
    # Target Log Transform
    target = np.log1p(train_df['SalePrice'])
    train_df = train_df.drop(['SalePrice', 'Id'], axis=1)
    test_df = test_df.drop(['Id'], axis=1)
    
    # Combine for consistent processing
    ntrain = train_df.shape[0]
    all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # --- Feature Creation ---
    
    # Fill NA for specific columns where NA means "None"
    for col in ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", 
                "GarageType", "GarageFinish", "GarageQual", "GarageCond", 
                "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", 
                "MasVnrType", "MSSubClass"]:
        all_data[col] = all_data[col].fillna("None")
        
    # Numerical imputation (Median is robust)
    num_cols = all_data.select_dtypes(include=np.number).columns
    for col in num_cols:
         all_data[col] = all_data[col].fillna(all_data[col].median())
         
    # Categorical imputation (Mode) - for others
    cat_cols = all_data.select_dtypes(include='object').columns
    for col in cat_cols:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # 1. Total Square Footage
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    
    # 2. Total Porch
    all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
                              all_data['WoodDeckSF'])
                              
    # 3. Ages
    # Fix YearBuilt if > YrSold
    all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
    all_data['HouseAge'] = all_data['HouseAge'].apply(lambda x: max(0, x))
    
    if 'GarageYrBlt' in all_data.columns:
         # Already imputed above, but ensure valid age
        all_data['GarageAge'] = all_data['YrSold'] - all_data['GarageYrBlt']
        all_data['GarageAge'] = all_data['GarageAge'].apply(lambda x: max(0, x))
    
    # Drop Utilities (almost constant)
    all_data = all_data.drop(['Utilities'], axis=1)

    # Dummies
    all_data = pd.get_dummies(all_data)
    
    print(f"Total Features: {all_data.shape[1]}")
    
    X = all_data[:ntrain]
    X_test = all_data[ntrain:]
    
    return X, target, X_test, test_ids

def train_models(X, y):
    print("\nTraining Models...")
    
    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling for Linear Models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(), # Alpha tuned later
        'Lasso': Lasso(alpha=0.0005), # Sensitive
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=3, n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        if name in ['LinearRegression', 'Ridge', 'Lasso']:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        results[name] = rmse
        print(f"{name} RMSE: {rmse:.5f}")
        
    return results, models, scaler

def optimize_best_model(X, y):
    print("\nOptimizing XGBoost (usually the best structural data performer)...")
    
    # Simple XGB hyperparameters
    model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
    
    # We will do a small Grid Search
    param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5],
        'colsample_bytree': [0.7, 1.0]
    }
    
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=1)
    grid.fit(X, y)
    
    print(f"Best RMSE from CV: {-grid.best_score_:.5f}")
    print(f"Best Params: {grid.best_params_}")
    
    return grid.best_estimator_

def main():
    try:
        train_df, test_df = load_data()
    except FileNotFoundError:
        print("Error: Could not find data files. Make sure they are in ../data/")
        return

    X, y, X_test, test_ids = feature_engineering(train_df, test_df)
    
    # 1. Baseline Comparison
    results, models, scaler = train_models(X, y)
    
    best_baseline = min(results, key=results.get)
    print(f"\nBest Baseline Model: {best_baseline} ({results[best_baseline]:.5f})")
    
    # 2. Optimization (Focusing on XGBoost as it's state of the art for this)
    final_model = optimize_best_model(X, y)
    
    # 3. Final Prediction
    print("\nGenerating Submission...")
    final_preds_log = final_model.predict(X_test)
    final_preds = np.expm1(final_preds_log) # Inverse Log
    
    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds})
    
    # Ensure outputs dir exists
    output_dir = 'outputs' if os.path.exists('outputs') else '../outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    submission.to_csv(f'{output_dir}/submission.csv', index=False)
    print(f"Saved {output_dir}/submission.csv")
    
    # Save Model
    model_dir = 'models' if os.path.exists('models') else '../models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    with open(f'{model_dir}/model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Saved {model_dir}/model.pkl")
    
    # Save Feature Names for App
    import json
    feature_names = list(X.columns)
    with open(f'{model_dir}/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    print(f"Saved {model_dir}/feature_names.json")

if __name__ == "__main__":
    main()
