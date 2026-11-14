
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, REPORTS_DIR

# Import FeatureCache from the v7 baseline script
from experiments.pooled_price_baseline_v7 import FeatureCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_feature_selection():
    logger.info("Starting Feature Selection for V7 Feature Set")
    
    # Setup paths
    output_dir = REPORTS_DIR / "feature_selection_v7"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    feature_cache = FeatureCache()
    
    # We found this file in the directory listing: features_v7_529stocks_2014-01-01_2024-12-31.npz
    # So n_stocks is likely 529 or similar. The load method asks for n_stocks.
    # However, the load method constructs the path.
    # We should probably list the files in the cache to find the exact one if we want to be robust,
    # or just try the one we saw.
    # But wait, the `load` method in `FeatureCache` takes `n_stocks`.
    # unique to that script is that it hardcodes checking if file exists.
    # Let's just manually load the file we saw to avoiding guessing n_stocks if it varies.
    
    cache_dir = DATA_DIR / "feature_cache"
    # Find the specific v7 file
    v7_files = list(cache_dir.glob("features_v7_*.npz"))
    if not v7_files:
        logger.error("No v7 feature cache file found!")
        return
    
    # Use the largest one or the first one
    cache_path = v7_files[0]
    logger.info(f"Loading data from: {cache_path}")
    
    data = np.load(cache_path, allow_pickle=True)
    X = data['X'] # (Samples, Sequence, Features)
    y = data['y']
    feature_cols = data['feature_cols'].tolist()
    
    logger.info(f"Loaded Data: X={X.shape}, y={y.shape}")
    logger.info(f"Number of original features: {len(feature_cols)}")
    
    # Prepare data for RF
    # Use the LAST time step of the sequence
    X_last = X[:, -1, :]
    logger.info(f"Flattened Data (Last Step): X={X_last.shape}")
    
    # Check for NaNs/Infs
    if np.any(np.isnan(X_last)) or np.any(np.isinf(X_last)):
        logger.warning("Data contains NaNs or Infs. Cleaning...")
        X_last = np.nan_to_num(X_last, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Run Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_last, y)
    
    # Get importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Save all features ranked
    ranking_df = pd.DataFrame({
        'Rank': range(1, len(feature_cols) + 1),
        'Feature': [feature_cols[i] for i in indices],
        'Importance': importances[indices]
    })
    
    ranking_path = output_dir / "feature_importance_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)
    logger.info(f"Saved feature ranking to {ranking_path}")
    
    # Select top 50
    top_50_features = ranking_df.head(50)['Feature'].tolist()
    
    # Save selected features list for future use
    selected_path = output_dir / "selected_features_v7_top50.txt"
    with open(selected_path, "w") as f:
        for feature in top_50_features:
            f.write(f"{feature}\n")
            
    logger.info(f"Saved top 50 features to {selected_path}")
    
    # Print top 10 for log
    logger.info("Top 10 Features:")
    print(ranking_df.head(10))

if __name__ == "__main__":
    run_feature_selection()
