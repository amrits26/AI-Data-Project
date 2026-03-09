#!/usr/bin/env python
"""Fix the corrupted SHAP section in modeling.py"""

import re

# Read the file
with open('backend/app/agents/modeling.py', 'r') as f:
    content = f.read()

# Define the corrupted pattern - it contains literal \n sequences as part of a patch diff
old_pattern = r'        # SHAP-style global importance.*?\n\s+except Exception as e:'

# Define the replacement (clean Python code)
new_code = '''        # SHAP-style global importance (mean |SHAP| per feature) for RandomForest
        shap_importance = None
        try:
            import shap

            # Use a small sample for efficiency
            sample_size = min(200, len(X_train))
            if sample_size > 0 and hasattr(model, "feature_importances_"):
                sample_X = X_train.sample(sample_size, random_state=42)
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(sample_X)
                if isinstance(shap_vals, list):
                    shap_arr = np.array(shap_vals[1 if len(shap_vals) > 1 else 0])
                else:
                    shap_arr = np.array(shap_vals)
                shap_abs_mean = np.mean(np.abs(shap_arr), axis=0)
                shap_importance = dict(
                    zip(
                        sample_X.columns.tolist(),
                        [float(x) for x in shap_abs_mean],
                    )
                )
        except Exception:
            shap_importance = None

        return {
            "problem_type": problem_type,
            "model_used": type(model).__name__,
            "cross_val_mean": float(np.mean(cv_scores)),
            "cross_val_std": float(np.std(cv_scores)),
            "cv_scores": [float(x) for x in cv_scores],
            "train_score": train_score,
            "test_score": test_score,
            "overfitting_analysis": overfit,
            "overfitting_risk": overfitting_risk_str,
            "inferred_task": problem_type,
            "summary": summary,
            "feature_importance": feat_imp,
            "best_model": type(model).__name__,
            "pred_vs_actual": pred_vs_actual,
            "residuals_sample": residuals_sample,
            "roc_curve": roc_curve_pts,
            "pr_curve": pr_curve_pts,
            "shap_importance": shap_importance,
        }

    except Exception as e:'''

# Use DOTALL flag to match across newlines
new_content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

# Write back
with open('backend/app/agents/modeling.py', 'w') as f:
    f.write(new_content)

print("✓ Fixed modeling.py - SHAP code corruption resolved")
