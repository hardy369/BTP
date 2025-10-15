import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BALANCED DATASET APPROACH FOR HEART ATTACK PREDICTION")
print("Handling Class Imbalance with Multiple Techniques")
print("="*80)

# ==================== LOAD DATA ====================

df = pd.read_csv('heart_attack_prediction_india (3).csv')
print(f"\nOriginal dataset: {df.shape}")

target_column = 'Heart_Attack_Risk'
exclude_cols = ['Patient_ID', target_column]
if 'State_Name' in df.columns:
    exclude_cols.append('State_Name')

X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
y = df[target_column]

# Handle categorical and missing values
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    X[col] = pd.Categorical(X[col]).codes
X = X.fillna(X.median())

print(f"\nOriginal class distribution:")
print(f"  Class 0 (No Risk): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"  Class 1 (At Risk): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"  Imbalance ratio: {(y==0).sum()/(y==1).sum():.2f}:1")

# ==================== TRAIN-TEST SPLIT ====================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)}, Test set: {len(X_test)}")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== RESAMPLING TECHNIQUES ====================

print("\n" + "="*80)
print("TESTING MULTIPLE RESAMPLING TECHNIQUES")
print("="*80)

resampling_methods = {
    'Original (Imbalanced)': None,
    'Random Oversampling': RandomOverSampler(random_state=42),
    'Random Undersampling': RandomUnderSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42, k_neighbors=5),
    'ADASYN': ADASYN(random_state=42),
    'SMOTE + Tomek': SMOTETomek(random_state=42),
    'SMOTE + ENN': SMOTEENN(random_state=42),
}

results_dict = {}

for method_name, sampler in resampling_methods.items():
    print(f"\n{'='*80}")
    print(f"Method: {method_name}")
    print(f"{'='*80}")
    
    # Apply resampling
    if sampler is None:
        X_resampled = X_train_scaled.copy()
        y_resampled = y_train.copy()
    else:
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
        except Exception as e:
            print(f"⚠️  Error with {method_name}: {e}")
            continue
    
    print(f"\nResampled distribution:")
    counter = Counter(y_resampled)
    print(f"  Class 0: {counter[0]} ({counter[0]/len(y_resampled)*100:.1f}%)")
    print(f"  Class 1: {counter[1]} ({counter[1]/len(y_resampled)*100:.1f}%)")
    print(f"  Total samples: {len(y_resampled)}")
    print(f"  Balance ratio: {counter[0]/counter[1]:.2f}:1")
    
    # Apply PCA on resampled data
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_resampled_pca = pca.fit_transform(X_resampled)
    X_test_pca = pca.transform(X_test_scaled)
    
    n_components = X_resampled_pca.shape[1]
    variance_explained = pca.explained_variance_ratio_.sum()
    
    print(f"\nPCA: {X_train_scaled.shape[1]} features → {n_components} components")
    print(f"Variance explained: {variance_explained*100:.2f}%")
    
    # Train Bayesian Logistic Regression (with L2 regularization)
    blr = LogisticRegression(
        penalty='l2',
        C=1.0,
        max_iter=2000,
        solver='lbfgs',
        random_state=42
    )
    
    blr.fit(X_resampled_pca, y_resampled)
    
    # Find optimal threshold
    y_train_proba = blr.predict_proba(X_resampled_pca)[:, 1]
    
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.3, 0.7, 0.01):
        y_pred_temp = (y_train_proba >= thresh).astype(int)
        f1 = f1_score(y_resampled, y_pred_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    print(f"Optimal threshold: {best_thresh:.2f}")
    
    # Predictions on test set
    y_test_proba = blr.predict_proba(X_test_pca)[:, 1]
    y_test_pred = (y_test_proba >= best_thresh).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {acc*100:6.2f}%")
    print(f"  Precision: {prec*100:6.2f}%")
    print(f"  Recall:    {rec*100:6.2f}%")
    print(f"  F1-Score:  {f1*100:6.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TP: {tp:4d}")
    print(f"  False Negative Rate: {fn/(fn+tp)*100:.2f}%")
    print(f"  False Positive Rate: {fp/(fp+tn)*100:.2f}%")
    
    # Store results
    results_dict[method_name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'n_samples': len(y_resampled),
        'n_components': n_components,
        'balance_ratio': counter[0]/counter[1],
        'model': blr,
        'pca': pca,
        'threshold': best_thresh
    }

# ==================== COMPARISON TABLE ====================

print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON OF ALL RESAMPLING METHODS")
print("="*80)

# Sort by F1-score
sorted_results = sorted(results_dict.items(), key=lambda x: x[1]['f1'], reverse=True)

print(f"\n{'Method':<25} {'Accuracy':>10} {'Precision':>11} {'Recall':>10} {'F1-Score':>10} {'FN':>6}")
print("-" * 82)

for method_name, metrics in sorted_results:
    marker = "🏆" if metrics['f1'] == max(r['f1'] for r in results_dict.values()) else "  "
    print(f"{marker} {method_name:<23} {metrics['accuracy']*100:>9.2f}% "
          f"{metrics['precision']*100:>10.2f}% {metrics['recall']*100:>9.2f}% "
          f"{metrics['f1']*100:>9.2f}% {metrics['fn']:>5d}")

# ==================== BEST METHOD ANALYSIS ====================

best_method_name = max(results_dict.items(), key=lambda x: x[1]['f1'])[0]
best_metrics = results_dict[best_method_name]

print("\n" + "="*80)
print(f"🏆 BEST METHOD: {best_method_name}")
print("="*80)

print(f"\nPerformance Metrics:")
print(f"  Accuracy:  {best_metrics['accuracy']*100:.2f}%")
print(f"  Precision: {best_metrics['precision']*100:.2f}%")
print(f"  Recall:    {best_metrics['recall']*100:.2f}%")
print(f"  F1-Score:  {best_metrics['f1']*100:.2f}%")

print(f"\nConfusion Matrix Breakdown:")
print(f"  True Negatives:  {best_metrics['tn']:4d} (Correct: No Risk)")
print(f"  False Positives: {best_metrics['fp']:4d} (False Alarm)")
print(f"  False Negatives: {best_metrics['fn']:4d} 🚨 (Missed Cases - CRITICAL)")
print(f"  True Positives:  {best_metrics['tp']:4d} (Correct: At Risk)")

print(f"\nKey Statistics:")
print(f"  False Negative Rate: {best_metrics['fn']/(best_metrics['fn']+best_metrics['tp'])*100:.2f}%")
print(f"  False Positive Rate: {best_metrics['fp']/(best_metrics['fp']+best_metrics['tn'])*100:.2f}%")
print(f"  Training samples: {best_metrics['n_samples']}")
print(f"  PCA components: {best_metrics['n_components']}")
print(f"  Classification threshold: {best_metrics['threshold']:.2f}")

# ==================== VISUALIZATIONS ====================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(16, 16))

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
methods = [m for m, _ in sorted_results]
accuracies = [results_dict[m]['accuracy']*100 for m in methods]
colors = ['gold' if m == best_method_name else 'skyblue' for m in methods]
bars = ax1.barh(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.axvline(x=70, color='red', linestyle='--', linewidth=2, alpha=0.7, label='70% Target')
ax1.set_xlabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax1.set_title('Accuracy Comparison', fontweight='bold', fontsize=14)
ax1.legend()
ax1.grid(axis='x', alpha=0.3)
for bar, acc in zip(bars, accuracies):
    ax1.text(acc+1, bar.get_y()+bar.get_height()/2, f'{acc:.1f}%', 
             va='center', fontweight='bold')

# Plot 2: F1-Score Comparison
ax2 = axes[0, 1]
f1_scores = [results_dict[m]['f1']*100 for m in methods]
colors2 = ['gold' if m == best_method_name else 'lightcoral' for m in methods]
bars2 = ax2.barh(methods, f1_scores, color=colors2, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('F1-Score (%)', fontweight='bold', fontsize=12)
ax2.set_title('F1-Score Comparison (Balanced Metric)', fontweight='bold', fontsize=14)
ax2.grid(axis='x', alpha=0.3)
for bar, f1 in zip(bars2, f1_scores):
    ax2.text(f1+1, bar.get_y()+bar.get_height()/2, f'{f1:.1f}%', 
             va='center', fontweight='bold')

# Plot 3: Recall Comparison
ax3 = axes[1, 0]
recalls = [results_dict[m]['recall']*100 for m in methods]
colors3 = ['gold' if m == best_method_name else 'lightgreen' for m in methods]
bars3 = ax3.barh(methods, recalls, color=colors3, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Recall (%) - Most Important for Medical!', fontweight='bold', fontsize=12)
ax3.set_title('Recall Comparison (Catching At-Risk Patients)', fontweight='bold', fontsize=14)
ax3.grid(axis='x', alpha=0.3)
for bar, rec in zip(bars3, recalls):
    ax3.text(rec+1, bar.get_y()+bar.get_height()/2, f'{rec:.1f}%', 
             va='center', fontweight='bold')

# Plot 4: False Negatives
ax4 = axes[1, 1]
false_negs = [results_dict[m]['fn'] for m in methods]
colors4 = ['gold' if m == best_method_name else 'salmon' for m in methods]
bars4 = ax4.barh(methods, false_negs, color=colors4, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('False Negatives (Lower is Better)', fontweight='bold', fontsize=12)
ax4.set_title('🚨 Missed Cases - CRITICAL METRIC', fontweight='bold', fontsize=14)
ax4.grid(axis='x', alpha=0.3)
for bar, fn in zip(bars4, false_negs):
    ax4.text(fn+5, bar.get_y()+bar.get_height()/2, f'{fn}', 
             va='center', fontweight='bold')

# Plot 5: Confusion Matrix for Best Method
ax5 = axes[2, 0]
cm_best = np.array([[best_metrics['tn'], best_metrics['fp']], 
                     [best_metrics['fn'], best_metrics['tp']]])
im = ax5.imshow(cm_best, cmap='RdYlGn_r', aspect='auto')
ax5.set_xticks([0, 1])
ax5.set_yticks([0, 1])
ax5.set_xticklabels(['Predicted:\nNo Risk', 'Predicted:\nAt Risk'], fontsize=11)
ax5.set_yticklabels(['Actual:\nNo Risk', 'Actual:\nAt Risk'], fontsize=11)
ax5.set_title(f'Confusion Matrix: {best_method_name}', fontweight='bold', fontsize=14)
for i in range(2):
    for j in range(2):
        text = ax5.text(j, i, f'{cm_best[i, j]}\n({cm_best[i,j]/cm_best.sum()*100:.1f}%)', 
                       ha="center", va="center",
                       color="white" if cm_best[i, j] > cm_best.max()/2 else "black",
                       fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax5)

# Plot 6: Precision-Recall Trade-off
ax6 = axes[2, 1]
precisions = [results_dict[m]['precision']*100 for m in methods]
recalls_plot = [results_dict[m]['recall']*100 for m in methods]
colors6 = ['gold' if m == best_method_name else 'purple' for m in methods]
for i, method in enumerate(methods):
    ax6.scatter(recalls_plot[i], precisions[i], s=300, color=colors6[i], 
               edgecolor='black', linewidth=2, alpha=0.7, zorder=3)
    if method == best_method_name:
        ax6.annotate(method, (recalls_plot[i], precisions[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.7))
ax6.set_xlabel('Recall (%)', fontweight='bold', fontsize=12)
ax6.set_ylabel('Precision (%)', fontweight='bold', fontsize=12)
ax6.set_title('Precision-Recall Trade-off', fontweight='bold', fontsize=14)
ax6.grid(alpha=0.3)
ax6.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax6.axvline(x=50, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('balanced_data_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: 'balanced_data_comprehensive_analysis.png'")

# ==================== RECOMMENDATIONS ====================

print("\n" + "="*80)
print("📋 RECOMMENDATIONS")
print("="*80)

print(f"\n1. BEST RESAMPLING METHOD: {best_method_name}")
print(f"   → Achieves best F1-Score: {best_metrics['f1']*100:.2f}%")
print(f"   → Balances precision and recall effectively")

if best_metrics['accuracy'] >= 0.70:
    print(f"\n2. ✅ ACCURACY TARGET MET: {best_metrics['accuracy']*100:.2f}%")
else:
    print(f"\n2. ⚠️  Accuracy: {best_metrics['accuracy']*100:.2f}% (Below 70% target)")
    print(f"   → However, F1-score and recall are more important for medical diagnosis")

print(f"\n3. MEDICAL PERSPECTIVE:")
print(f"   → False Negatives: {best_metrics['fn']} patients (missed cases)")
print(f"   → False Positives: {best_metrics['fp']} patients (false alarms)")
print(f"   → For medical diagnosis, minimizing false negatives is CRITICAL")
print(f"   → Current model catches {best_metrics['recall']*100:.1f}% of at-risk patients")

print(f"\n4. DEPLOYMENT RECOMMENDATION:")
if best_metrics['recall'] >= 0.80 and best_metrics['f1'] >= 0.50:
    print(f"   ✅ This model is suitable for deployment")
    print(f"   → High recall ensures most at-risk patients are identified")
    print(f"   → Acceptable precision reduces unnecessary follow-ups")
else:
    print(f"   ⚠️  Consider further tuning:")
    print(f"   → Adjust classification threshold for better balance")
    print(f"   → Collect more data for minority class")
    print(f"   → Try ensemble methods")

# ==================== SAVE BEST MODEL RESULTS ====================

# Use best model for final predictions
best_model = results_dict[best_method_name]['model']
best_pca = results_dict[best_method_name]['pca']
best_threshold = results_dict[best_method_name]['threshold']

X_test_pca_final = best_pca.transform(X_test_scaled)
y_test_proba_final = best_model.predict_proba(X_test_pca_final)[:, 1]
y_test_pred_final = (y_test_proba_final >= best_threshold).astype(int)

output_df = pd.DataFrame({
    'True_Label': y_test.values,
    'Predicted_Label': y_test_pred_final,
    'Probability_At_Risk': y_test_proba_final,
    'Correct': (y_test.values == y_test_pred_final),
    'Risk_Category': ['True Positive' if (t==1 and p==1) else 
                     'True Negative' if (t==0 and p==0) else
                     'False Positive' if (t==0 and p==1) else 
                     'False Negative' 
                     for t, p in zip(y_test.values, y_test_pred_final)]
})

output_df.to_csv('best_balanced_model_results.csv', index=False)
print(f"\n✓ Best model predictions saved: 'best_balanced_model_results.csv'")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"\nBest Method: {best_method_name}")
print(f"Test Accuracy: {best_metrics['accuracy']*100:.2f}%")
print(f"Test F1-Score: {best_metrics['f1']*100:.2f}%")
print(f"Test Recall: {best_metrics['recall']*100:.2f}%")
print("="*80)