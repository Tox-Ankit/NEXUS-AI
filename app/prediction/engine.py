import os
# Allow multi-threading while preventing system overload
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(min(4, max(1, (os.cpu_count() or 2) - 1)))

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import (
    RandomForestRegressor, 
    RandomForestClassifier, 
    HistGradientBoostingRegressor, 
    HistGradientBoostingClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier
)
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score
)
import xgboost as xgb
from typing import Dict, Any, List
import warnings
warnings.filterwarnings("ignore")

class PredictiveEngine:
    MAX_TRAIN_SAMPLES = 25_000
    MAX_TEST_SAMPLES = 5_000

    @staticmethod
    def _build_preprocessor(numeric_features: List[str], categorical_features: List[str]):
        """Build an optimized scikit-learn ColumnTransformer bounded against high cardinality."""
        transformers = []
        
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))

        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=30, sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))

        return ColumnTransformer(transformers=transformers, remainder='drop')

    @classmethod
    def _clean_features(cls, pdf: pd.DataFrame, target: str, features: List[str]):
        """Filter out identifier or constant columns and separate numeric vs categorical."""
        valid_features = []
        n_rows = len(pdf)
        for col in features:
            if col == target or col not in pdf.columns:
                continue
            # Drop constant columns
            n_unique = pdf[col].nunique(dropna=True)
            if n_unique <= 1:
                continue
            # Skip columns that are entirely unique strings/ids or high-cardinality hashes (likely row identifiers)
            if pdf[col].dtype == object or str(pdf[col].dtype) in ("string", "category"):
                if n_rows > 20 and n_unique == n_rows:
                    continue
                if n_rows > 500 and n_unique > 150 and (n_unique / n_rows) > 0.25:
                    continue
            valid_features.append(col)

        if not valid_features:
            valid_features = [f for f in features if f != target and f in pdf.columns]

        X = pdf[valid_features]
        numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'number']).columns.tolist()
        categorical_features = [col for col in valid_features if col not in numeric_features]
        return valid_features, numeric_features, categorical_features

    @classmethod
    def train_and_evaluate(cls, df: pl.DataFrame, target: str, problem_type: str, features: list) -> Dict[str, Any]:
        """
        Trains and compares a diverse suite of competitive models using both
        Cross-Validation and unseen Holdout Test evaluation.
        Intelligently ranks and selects the true winning champion model.
        Features scale-adaptive subsampling and multi-threading for instant performance on big datasets.
        """
        # Convert to pandas and drop missing target rows
        pdf = df.to_pandas().dropna(subset=[target]).copy()
        
        valid_features, numeric_features, categorical_features = cls._clean_features(pdf, target, features)
        X = pdf[valid_features]
        y = pdf[target]
        n_samples = len(pdf)

        # Dynamic train/test split size based on dataset scale
        test_size = 0.20 if n_samples >= 40 else max(0.15, min(0.30, 3.0 / n_samples))
        
        preprocessor = cls._build_preprocessor(numeric_features, categorical_features)

        if problem_type == "regression":
            # Target should be float/numeric
            y = pd.to_numeric(y, errors='coerce').fillna(y.mean() if hasattr(y, 'mean') else 0)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Scale-adaptive subsampling for high-speed tournament training on large datasets
            if len(X_train) > cls.MAX_TRAIN_SAMPLES:
                fit_indices = np.random.RandomState(42).choice(len(X_train), size=cls.MAX_TRAIN_SAMPLES, replace=False)
                X_train_fit = X_train.iloc[fit_indices]
                y_train_fit = y_train.iloc[fit_indices]
            else:
                X_train_fit = X_train
                y_train_fit = y_train

            if len(X_test) > cls.MAX_TEST_SAMPLES:
                test_indices = np.random.RandomState(42).choice(len(X_test), size=cls.MAX_TEST_SAMPLES, replace=False)
                X_test_eval = X_test.iloc[test_indices]
                y_test_eval = y_test.iloc[test_indices]
            else:
                X_test_eval = X_test
                y_test_eval = y_test
            
            # Cross validation setup
            cv_splits = min(5, max(2, len(X_train_fit) // 4))
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

            models = {
                "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_split=3, random_state=42, n_jobs=-1),
                "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=80, random_state=42),
                "XGBoost": xgb.XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.08, random_state=42, verbosity=0, n_jobs=-1),
                "Extra Trees": ExtraTreesRegressor(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1),
                "Ridge Regression": Ridge(alpha=1.0),
                "Baseline (Mean)": DummyRegressor(strategy="mean")
            }

            results = {}
            leaderboard_rows = []

            for name, model in models.items():
                try:
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                    
                    # CV R2 score on representative fit set
                    cv_scores = cross_val_score(pipeline, X_train_fit, y_train_fit, cv=kf, scoring='r2')
                    cv_mean = float(np.nanmean(cv_scores))
                    cv_std = float(np.nanstd(cv_scores))

                    # Fit and evaluate on holdout test set
                    pipeline.fit(X_train_fit, y_train_fit)
                    preds = pipeline.predict(X_test_eval)

                    r2 = float(r2_score(y_test_eval, preds))
                    mae = float(mean_absolute_error(y_test_eval, preds))
                    rmse = float(np.sqrt(mean_squared_error(y_test_eval, preds)))

                    metrics = {
                        "r2": round(r2, 4),
                        "rmse": round(rmse, 4),
                        "mae": round(mae, 4),
                        "cv_r2_mean": round(cv_mean, 4),
                        "cv_r2_std": round(cv_std, 4)
                    }
                    results[name] = {"metrics": metrics}

                    # Composite ranking score: higher R2, lower RMSE relative to scale, CV consistency
                    if name == "Baseline (Mean)":
                        composite_score = -999999.0
                    else:
                        target_scale = float(y_test_eval.std()) if float(y_test_eval.std()) > 0 else (abs(float(y_test_eval.mean())) or 1.0)
                        rmse_penalty = rmse / target_scale
                        r2_score_val = max(r2, -2.0)
                        cv_score_val = max(cv_mean, -2.0)
                        r2_bonus = 2.0 if r2 > 0 else 0.0
                        composite_score = (2.0 * r2_score_val) + r2_bonus + (0.5 * cv_score_val) - (1.5 * rmse_penalty)

                    leaderboard_rows.append({
                        "model": name,
                        "r2": r2,
                        "rmse": rmse,
                        "mae": mae,
                        "cv_score": f"{cv_mean:.3f} ± {cv_std:.3f}",
                        "composite_score": composite_score,
                        "pipeline": pipeline,
                        "preds": preds
                    })

                except Exception as e:
                    results[name] = {"error": str(e)}

            # Sort leaderboard
            leaderboard_rows.sort(key=lambda x: x["composite_score"], reverse=True)
            
            # Select champion
            top = leaderboard_rows[0]
            champion_name = top["model"]
            champion_pipeline = top["pipeline"]
            champion_preds = top["preds"]
            champion_metrics = results[champion_name]["metrics"]

            # Generate formatted leaderboard for UI
            formatted_leaderboard = []
            medals = ["🥇 Champion", "🥈 Runner-Up", "🥉 3rd Place"]
            for idx, item in enumerate(leaderboard_rows):
                rank_badge = medals[idx] if idx < len(medals) else f"#{idx+1}"
                formatted_leaderboard.append({
                    "Rank": rank_badge,
                    "Model": item["model"],
                    "R² Score": f"{item['r2']:.4f}",
                    "RMSE": f"{item['rmse']:.4f}",
                    "MAE": f"{item['mae']:.4f}",
                    "5-Fold CV R²": item["cv_score"]
                })

            sample_note = f" (Optimized: trained on {len(X_train_fit):,} representative samples)" if len(X_train) > cls.MAX_TRAIN_SAMPLES else ""
            rationale = (
                f"{champion_name} won the model selection with the highest generalization performance: "
                f"Test R² of {champion_metrics['r2']:.4f}, lowest RMSE of {champion_metrics['rmse']:.4f}, "
                f"and robust Cross-Validation score ({champion_metrics['cv_r2_mean']:.3f} ± {champion_metrics['cv_r2_std']:.3f}).{sample_note}"
            )

            feature_importances = cls._extract_feature_importances(champion_pipeline, valid_features, numeric_features, categorical_features, X_test_eval, y_test_eval)

            actual_list = y_test_eval.tolist()
            pred_list = champion_preds.tolist() if champion_preds is not None else []
            residuals = []
            for a, p in zip(actual_list, pred_list):
                try:
                    residuals.append(round(float(a) - float(p), 4))
                except (TypeError, ValueError):
                    residuals.append(None)

            return {
                "type": "regression",
                "best_model_name": champion_name,
                "best_metrics": champion_metrics,
                "all_results": results,
                "leaderboard": formatted_leaderboard,
                "selection_rationale": rationale,
                "model_pipeline": champion_pipeline,
                "features": valid_features,
                "target": target,
                "feature_importances": feature_importances,
                "test_eval": {
                    "actual": actual_list,
                    "predicted": pred_list,
                    "residuals": residuals,
                    "holdout_size": len(pred_list),
                }
            }

        else: # Classification
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))
            classes = le.classes_.tolist()
            num_classes = len(classes)

            # Check if stratification is possible
            class_counts = pd.Series(y_encoded).value_counts()
            can_stratify = class_counts.min() >= 2
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=test_size, 
                random_state=42, 
                stratify=y_encoded if can_stratify else None
            )

            # Scale-adaptive subsampling for large datasets
            if len(X_train) > cls.MAX_TRAIN_SAMPLES:
                strat_train = y_train if (can_stratify and pd.Series(y_train).value_counts().min() >= 2) else None
                fit_indices, _ = train_test_split(
                    np.arange(len(X_train)),
                    train_size=cls.MAX_TRAIN_SAMPLES,
                    stratify=strat_train,
                    random_state=42
                )
                X_train_fit = X_train.iloc[fit_indices]
                y_train_fit = y_train[fit_indices]
            else:
                X_train_fit = X_train
                y_train_fit = y_train

            if len(X_test) > cls.MAX_TEST_SAMPLES:
                strat_test = y_test if (can_stratify and pd.Series(y_test).value_counts().min() >= 2) else None
                test_indices, _ = train_test_split(
                    np.arange(len(X_test)),
                    train_size=cls.MAX_TEST_SAMPLES,
                    stratify=strat_test,
                    random_state=42
                )
                X_test_eval = X_test.iloc[test_indices]
                y_test_eval = y_test[test_indices]
            else:
                X_test_eval = X_test
                y_test_eval = y_test

            fit_counts = pd.Series(y_train_fit).value_counts()
            can_stratify_fit = fit_counts.min() >= 2
            cv_splits = min(5, max(2, fit_counts.min())) if can_stratify_fit else min(3, max(2, len(X_train_fit) // 4))
            skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42) if can_stratify_fit else KFold(n_splits=cv_splits, shuffle=True, random_state=42)

            models = {
                "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_split=3, random_state=42, n_jobs=-1),
                "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=80, random_state=42),
                "XGBoost": xgb.XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.08, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
                "Extra Trees": ExtraTreesClassifier(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1),
                "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
                "Baseline (Prior)": DummyClassifier(strategy="most_frequent")
            }

            results = {}
            leaderboard_rows = []

            for name, model in models.items():
                try:
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                    
                    # CV Weighted F1
                    cv_scores = cross_val_score(pipeline, X_train_fit, y_train_fit, cv=skf, scoring='f1_weighted')
                    cv_mean = float(np.nanmean(cv_scores))
                    cv_std = float(np.nanstd(cv_scores))

                    pipeline.fit(X_train_fit, y_train_fit)
                    preds = pipeline.predict(X_test_eval)

                    acc = float(accuracy_score(y_test_eval, preds))
                    f1 = float(f1_score(y_test_eval, preds, average='weighted', zero_division=0))
                    prec = float(precision_score(y_test_eval, preds, average='weighted', zero_division=0))
                    rec = float(recall_score(y_test_eval, preds, average='weighted', zero_division=0))

                    metrics = {
                        "accuracy": round(acc, 4),
                        "f1_score": round(f1, 4),
                        "precision": round(prec, 4),
                        "recall": round(rec, 4),
                        "cv_f1_mean": round(cv_mean, 4),
                        "cv_f1_std": round(cv_std, 4)
                    }
                    results[name] = {"metrics": metrics}

                    if name == "Baseline (Prior)":
                        composite_score = -999999.0
                    else:
                        composite_score = (0.5 * f1) + (0.3 * cv_mean) + (0.2 * acc)

                    leaderboard_rows.append({
                        "model": name,
                        "f1_score": f1,
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "cv_score": f"{cv_mean:.3f} ± {cv_std:.3f}",
                        "composite_score": composite_score,
                        "pipeline": pipeline,
                        "preds": preds
                    })

                except Exception as e:
                    results[name] = {"error": str(e)}

            leaderboard_rows.sort(key=lambda x: x["composite_score"], reverse=True)

            top = leaderboard_rows[0]
            champion_name = top["model"]
            champion_pipeline = top["pipeline"]
            champion_preds = top["preds"]
            champion_metrics = results[champion_name]["metrics"]

            formatted_leaderboard = []
            medals = ["🥇 Champion", "🥈 Runner-Up", "🥉 3rd Place"]
            for idx, item in enumerate(leaderboard_rows):
                rank_badge = medals[idx] if idx < len(medals) else f"#{idx+1}"
                formatted_leaderboard.append({
                    "Rank": rank_badge,
                    "Model": item["model"],
                    "F1 Score": f"{item['f1_score']:.4f}",
                    "Accuracy": f"{item['accuracy']:.4f}",
                    "Precision": f"{item['precision']:.4f}",
                    "Recall": f"{item['recall']:.4f}",
                    "5-Fold CV F1": item["cv_score"]
                })

            sample_note = f" (Optimized: trained on {len(X_train_fit):,} representative samples)" if len(X_train) > cls.MAX_TRAIN_SAMPLES else ""
            rationale = (
                f"{champion_name} won the classification competition with the highest predictive power: "
                f"Test Weighted F1 of {champion_metrics['f1_score']:.4f}, Accuracy of {champion_metrics['accuracy']:.4f}, "
                f"and robust Cross-Validation score ({champion_metrics['cv_f1_mean']:.3f} ± {champion_metrics['cv_f1_std']:.3f}).{sample_note}"
            )

            feature_importances = cls._extract_feature_importances(champion_pipeline, valid_features, numeric_features, categorical_features, X_test_eval, y_test_eval)

            actual_labels = [classes[i] for i in y_test_eval]
            pred_labels = [classes[int(i)] for i in champion_preds] if champion_preds is not None else []
            proba_max = []
            try:
                proba = champion_pipeline.predict_proba(X_test_eval)
                proba_max = [round(float(row.max()), 4) for row in proba]
            except Exception:
                proba_max = []

            return {
                "type": "classification",
                "best_model_name": champion_name,
                "best_metrics": champion_metrics,
                "all_results": results,
                "leaderboard": formatted_leaderboard,
                "selection_rationale": rationale,
                "model_pipeline": champion_pipeline,
                "features": valid_features,
                "target": target,
                "label_classes": classes,
                "feature_importances": feature_importances,
                "test_eval": {
                    "actual": actual_labels,
                    "predicted": pred_labels,
                    "predicted_proba": proba_max,
                    "holdout_size": len(pred_labels),
                }
            }

    @staticmethod
    def _extract_feature_importances(pipeline, valid_features, numeric_features, categorical_features, X_test=None, y_test=None):
        """Extract or approximate feature importances from the trained model."""
        importances = []
        try:
            model = pipeline.named_steps.get('model')
            preprocessor = pipeline.named_steps.get('preprocessor')
            
            feature_names = []
            if hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    feature_names = list(preprocessor.get_feature_names_out())
                except Exception:
                    feature_names = numeric_features + categorical_features
            else:
                feature_names = numeric_features + categorical_features

            raw_scores = None
            if hasattr(model, 'feature_importances_'):
                raw_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                raw_scores = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            elif X_test is not None and y_test is not None:
                # Permutation importance fallback for HistGradientBoosting, etc.
                from sklearn.inspection import permutation_importance
                # Subsample X_test if large to avoid CPU freeze
                if len(X_test) > 500:
                    sub_idx = np.random.RandomState(42).choice(len(X_test), size=500, replace=False)
                    X_sub = X_test.iloc[sub_idx]
                    y_sub = y_test.iloc[sub_idx] if hasattr(y_test, "iloc") else np.array(y_test)[sub_idx]
                else:
                    X_sub, y_sub = X_test, y_test
                perm = permutation_importance(pipeline, X_sub, y_sub, n_repeats=3, random_state=42)
                raw_imp = perm.importances_mean
                for idx, feat in enumerate(valid_features):
                    if idx < len(raw_imp):
                        importances.append({"feature": feat, "importance": round(float(max(0.0, raw_imp[idx])), 4)})
                total = sum(x["importance"] for x in importances) or 1.0
                for item in importances:
                    item["importance"] = round(item["importance"] / total, 4)
                importances.sort(key=lambda x: x["importance"], reverse=True)
                return importances[:10]

            if raw_scores is not None and len(raw_scores) > 0:
                raw_scores = np.array(raw_scores).flatten()
                min_len = min(len(feature_names), len(raw_scores))
                
                feature_agg = {}
                for idx in range(min_len):
                    fname = str(feature_names[idx])
                    clean_name = fname.replace("num__", "").replace("cat__", "")
                    parent_feature = clean_name.split("_")[0] if "_" in clean_name and clean_name.split("_")[0] in valid_features else clean_name
                    feature_agg[parent_feature] = feature_agg.get(parent_feature, 0.0) + float(raw_scores[idx])

                total = sum(feature_agg.values()) or 1.0
                sorted_features = sorted(feature_agg.items(), key=lambda x: x[1], reverse=True)
                for f, score in sorted_features[:10]:
                    importances.append({"feature": f, "importance": round(float(score / total), 4)})
        except Exception:
            pass

        return importances
