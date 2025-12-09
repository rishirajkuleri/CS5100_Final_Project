"""
CS5100 Phase 2 Implementation
Solo student implementation including:
1. Full dataset usage (1 point)
2. Feature selection (1 point) 
3. Stacking ensemble (2 points)

"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
import warnings
warnings.filterwarnings('ignore')

# Import Phase 1 implementations
from student_project.student_project import (
    load_data, preprocess_data, train_gb_pipeline, RandomForest
)


class Phase2Implementation:
    """
    Phase 2 implementation with advanced techniques.
    """
    
    def __init__(self, use_full_dataset=True, random_state=42):
        """
        Initialize Phase 2 implementation.
        
        Parameters:
        - use_full_dataset: If True, use full student-mat.csv; else use mini
        - random_state: Random seed for reproducibility
        """
        self.use_full_dataset = use_full_dataset
        self.random_state = random_state
        self.results = {}
        
    def load_and_preprocess_data(self):
        """
        Load dataset and preprocess.
        Returns processed DataFrame split into train/test.
        """
        if self.use_full_dataset:
            path = 'datasets/student-mat.csv'
            print(f"Loading full dataset from {path}")
        else:
            path = None
            print("Loading mini dataset")
        
        df = load_data(path) if self.use_full_dataset else load_data()
        print(f"Dataset shape: {df.shape}")
        print(f"At-risk ratio: {(df['G3'] < 10).sum()} / {len(df)} = {(df['G3'] < 10).mean():.2%}")

        processed = preprocess_data(df)

        X = processed.drop(columns=['at_risk'])
        y = processed['at_risk']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Train at-risk: {y_train.sum()} ({y_train.mean():.2%})")
        print(f"Test at-risk: {y_test.sum()} ({y_test.mean():.2%})")
        
        return X_train, X_test, y_train, y_test
    
    def baseline_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate baseline models from Phase 1.
        """
        print("BASELINE MODELS (Phase 1)")
        
        results = {}

        print("\n1. Gradient Boosting Classifier")
        gb_model = train_gb_pipeline(X_train, y_train)
        y_pred_gb = gb_model.predict(X_test)
        
        f1_gb = f1_score(y_test, y_pred_gb, zero_division=0)
        acc_gb = accuracy_score(y_test, y_pred_gb)
        
        try:
            y_prob_gb = gb_model.predict_proba(X_test)[:, 1]
            auc_gb = roc_auc_score(y_test, y_prob_gb)
        except:
            auc_gb = 0.0
        
        print(f"   F1 Score: {f1_gb:.4f}")
        print(f"   Accuracy: {acc_gb:.4f}")
        print(f"   ROC-AUC:  {auc_gb:.4f}")
        
        results['Gradient Boosting'] = {
            'model': gb_model,
            'f1': f1_gb,
            'accuracy': acc_gb,
            'auc': auc_gb
        }

        print("\n2. Random Forest (From Scratch)")
        rf = RandomForest(
            n_estimators=15, 
            max_depth=6, 
            sample_size=min(200, len(X_train)),
            random_state=self.random_state
        )
        rf.fit(X_train.values, y_train.values)
        y_pred_rf = rf.predict(X_test.values)
        
        f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        
        print(f"   F1 Score: {f1_rf:.4f}")
        print(f"   Accuracy: {acc_rf:.4f}")
        
        results['Random Forest'] = {
            'model': rf,
            'f1': f1_rf,
            'accuracy': acc_rf,
            'auc': 0.0 
        }
        
        self.results['baseline'] = results
        return results
    
    def feature_selection(self, X_train, X_test, y_train, y_test):
        """
        Apply feature selection to reduce dimensionality.
        Tests multiple methods and selects best performing.
        """
        print("FEATURE SELECTION")
        
        print(f"Original feature count: {X_train.shape[1]}")
        
        results = {}

        print("\n1. Mutual Information Feature Selection")
        k_best = min(20, X_train.shape[1])
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=k_best)
        X_train_mi = selector_mi.fit_transform(X_train, y_train)
        X_test_mi = selector_mi.transform(X_test)

        selected_features_mi = X_train.columns[selector_mi.get_support()].tolist()
        print(f"Selected {len(selected_features_mi)} features")
        print(f"Top 5: {selected_features_mi[:5]}")

        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=self.random_state
        )
        gb.fit(X_train_mi, y_train)
        y_pred_mi = gb.predict(X_test_mi)
        
        f1_mi = f1_score(y_test, y_pred_mi, zero_division=0)
        acc_mi = accuracy_score(y_test, y_pred_mi)
        auc_mi = roc_auc_score(y_test, gb.predict_proba(X_test_mi)[:, 1])
        
        print(f"F1 Score: {f1_mi:.4f}")
        print(f"Accuracy: {acc_mi:.4f}")
        print(f"ROC-AUC:  {auc_mi:.4f}")
        
        results['Mutual Information'] = {
            'selector': selector_mi,
            'features': selected_features_mi,
            'X_train': X_train_mi,
            'X_test': X_test_mi,
            'f1': f1_mi,
            'accuracy': acc_mi,
            'auc': auc_mi
        }

        print("\n2. Recursive Feature Elimination (RFE)")
        n_features_rfe = min(15, X_train.shape[1])
        base_estimator = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=self.random_state
        )
        selector_rfe = RFE(estimator=base_estimator, n_features_to_select=n_features_rfe)
        X_train_rfe = selector_rfe.fit_transform(X_train, y_train)
        X_test_rfe = selector_rfe.transform(X_test)
        
        selected_features_rfe = X_train.columns[selector_rfe.get_support()].tolist()
        print(f"Selected {len(selected_features_rfe)} features")
        print(f"Top 5: {selected_features_rfe[:5]}")

        gb_rfe = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=self.random_state
        )
        gb_rfe.fit(X_train_rfe, y_train)
        y_pred_rfe = gb_rfe.predict(X_test_rfe)
        
        f1_rfe = f1_score(y_test, y_pred_rfe, zero_division=0)
        acc_rfe = accuracy_score(y_test, y_pred_rfe)
        auc_rfe = roc_auc_score(y_test, gb_rfe.predict_proba(X_test_rfe)[:, 1])
        
        print(f"   F1 Score: {f1_rfe:.4f}")
        print(f"   Accuracy: {acc_rfe:.4f}")
        print(f"   ROC-AUC:  {auc_rfe:.4f}")
        
        results['RFE'] = {
            'selector': selector_rfe,
            'features': selected_features_rfe,
            'X_train': X_train_rfe,
            'X_test': X_test_rfe,
            'f1': f1_rfe,
            'accuracy': acc_rfe,
            'auc': auc_rfe
        }

        best_method = max(results.keys(), key=lambda k: results[k]['f1'])
        print(f"\nBest feature selection method: {best_method}")
        print(f"F1 improvement over baseline: {results[best_method]['f1'] - self.results['baseline']['Gradient Boosting']['f1']:.4f}")
        
        self.results['feature_selection'] = results
        return results[best_method]
    
    def stacking_ensemble(self, X_train, X_test, y_train, y_test, 
                          X_train_fs=None, X_test_fs=None):
        """
        Implement stacking ensemble combining multiple base models.
        Uses Gradient Boosting, Logistic Regression, and optionally feature-selected models.
        """
        print("STACKING ENSEMBLE")

        base_models = [
            ('gb1', GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, 
                random_state=self.random_state
            )),
            ('gb2', GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                random_state=self.random_state + 1
            )),
            ('lr', LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ))
        ]

        meta_model = LogisticRegression(
            max_iter=1000, random_state=self.random_state
        )
        
        print(f"\nBase models: {[name for name, _ in base_models]}")
        print(f"MetaModel: LogisticRegression")

        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )

        print("\nTraining stacking ensemble")
        stacking_clf.fit(X_train, y_train)

        y_pred_stack = stacking_clf.predict(X_test)
        y_prob_stack = stacking_clf.predict_proba(X_test)[:, 1]
        
        f1_stack = f1_score(y_test, y_pred_stack, zero_division=0)
        acc_stack = accuracy_score(y_test, y_pred_stack)
        auc_stack = roc_auc_score(y_test, y_prob_stack)
        
        print(f"\nStacking Ensemble Results:")
        print(f"F1 Score: {f1_stack:.4f}")
        print(f"Accuracy: {acc_stack:.4f}")
        print(f"ROC-AUC:  {auc_stack:.4f}")

        baseline_best_f1 = max(
            self.results['baseline']['Gradient Boosting']['f1'],
            self.results['baseline']['Random Forest']['f1']
        )
        improvement = f1_stack - baseline_best_f1
        
        print(f"\nImprovement over best baseline: {improvement:+.4f}")
        if improvement > 0:
            print("Stacking ensemble outperforms baseline models")
        
        results = {
            'model': stacking_clf,
            'f1': f1_stack,
            'accuracy': acc_stack,
            'auc': auc_stack,
            'improvement': improvement
        }

        if X_train_fs is not None and X_test_fs is not None:
            print("\nStacking with Feature Selection")
            
            stacking_clf_fs = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1
            )
            
            stacking_clf_fs.fit(X_train_fs, y_train)
            y_pred_stack_fs = stacking_clf_fs.predict(X_test_fs)
            y_prob_stack_fs = stacking_clf_fs.predict_proba(X_test_fs)[:, 1]
            
            f1_stack_fs = f1_score(y_test, y_pred_stack_fs, zero_division=0)
            acc_stack_fs = accuracy_score(y_test, y_pred_stack_fs)
            auc_stack_fs = roc_auc_score(y_test, y_prob_stack_fs)
            
            print(f"F1 Score: {f1_stack_fs:.4f}")
            print(f"Accuracy: {acc_stack_fs:.4f}")
            print(f"ROC-AUC:  {auc_stack_fs:.4f}")
            print(f"Improvement over baseline: {f1_stack_fs - baseline_best_f1:+.4f}")
            
            results['feature_selected'] = {
                'model': stacking_clf_fs,
                'f1': f1_stack_fs,
                'accuracy': acc_stack_fs,
                'auc': auc_stack_fs
            }
        
        self.results['stacking'] = results
        return results
    
    def run_full_pipeline(self):
        """
        Execute complete Phase 2 pipeline.
        """
        print("CS5100 PHASE 2")         

        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()        
        baseline_results = self.baseline_models(X_train, X_test, y_train, y_test)
        fs_results = self.feature_selection(X_train, X_test, y_train, y_test)
        stacking_results = self.stacking_ensemble(
            X_train, X_test, y_train, y_test,
            fs_results['X_train'], fs_results['X_test']
        )
        self.print_summary()        
        return self.results
    
    def print_summary(self):
        """
        Print comprehensive results summary.
        """
        print("PHASE 2 RESULTS")       
        print("\nBaseline Models (Phase 1):")
        for name, res in self.results['baseline'].items():
            print(f"  {name:25s} F1: {res['f1']:.4f}  Acc: {res['accuracy']:.4f}  AUC: {res['auc']:.4f}")
        
        print("\nFeature Selection:")
        for name, res in self.results['feature_selection'].items():
            print(f"  {name:25s} F1: {res['f1']:.4f}  Acc: {res['accuracy']:.4f}  AUC: {res['auc']:.4f}")
        
        print("\nStacking Ensemble:")
        res = self.results['stacking']
        print(f"  {'Standard':25s} F1: {res['f1']:.4f}  Acc: {res['accuracy']:.4f}  AUC: {res['auc']:.4f}")
        if 'feature_selected' in res:
            fs_res = res['feature_selected']
            print(f"  {'With Feature Selection':25s} F1: {fs_res['f1']:.4f}  Acc: {fs_res['accuracy']:.4f}  AUC: {fs_res['auc']:.4f}")

        all_f1s = {
            'GB Baseline': self.results['baseline']['Gradient Boosting']['f1'],
            'RF Baseline': self.results['baseline']['Random Forest']['f1'],
            'Stacking': self.results['stacking']['f1']
        }
        
        best_model = max(all_f1s.keys(), key=lambda k: all_f1s[k])
        best_f1 = all_f1s[best_model]

        print(f"BEST MODEL: {best_model}")
        print(f"F1 Score: {best_f1:.4f}")

def main():
    """
    Main execution function.
    """
    phase2 = Phase2Implementation(use_full_dataset=True, random_state=42)
    results = phase2.run_full_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()
