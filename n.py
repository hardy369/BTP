"""
Bayesian Logistic Regression for Heart Attack Risk Prediction
Based on: Hassan, M.M. (2020). A Fully Bayesian Logistic Regression Model 
for Classification of ZADA Diabetes Dataset

Alternative implementation using NumPyro (JAX-based) to avoid C compilation issues
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set NumPyro platform
numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

class BayesianLogisticRegression:
    """
    Fully Bayesian Logistic Regression using MCMC (NUTS sampler with NumPyro)
    Implements three different priors: Gaussian, Laplace, and Cauchy
    """
    
    def __init__(self, prior_type='gaussian', prior_params=None, standardize=True):
        """
        Initialize BLR model
        
        Parameters:
        -----------
        prior_type : str, {'gaussian', 'laplace', 'cauchy'}
            Type of prior distribution for coefficients
        prior_params : dict
            Parameters for prior distribution
        standardize : bool
            Whether to standardize features (default: True)
        """
        self.prior_type = prior_type.lower()
        self.prior_params = prior_params or self._default_prior_params()
        self.mcmc = None
        self.samples = None
        self.standardize = standardize
        if standardize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        
    def _default_prior_params(self):
        """Set default prior parameters based on paper (CORRECTED)"""
        defaults = {
            # Paper used N(0, 10) but on standardized scale
            # For unstandardized, use tighter priors
            'gaussian': {'mu': 0.0, 'sigma': 2.5},  # Informative, not too wide
            'laplace': {'mu': 0.0, 'scale': 1.0},   # Weakly informative
            'cauchy': {'loc': 0.0, 'scale': 2.5}    # Weakly informative (not 0.01!)
        }
        return defaults[self.prior_type]
    
    def _model(self, X, y=None):
        """
        NumPyro model definition for Bayesian Logistic Regression
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like, optional
            Labels
        """
        n_features = X.shape[1]
        
        # Define priors for intercept
        if self.prior_type == 'gaussian':
            intercept = numpyro.sample('intercept', 
                                      dist.Normal(self.prior_params['mu'], 
                                                 self.prior_params['sigma']))
            beta = numpyro.sample('beta', 
                                 dist.Normal(self.prior_params['mu'], 
                                           self.prior_params['sigma']).expand([n_features]))
        
        elif self.prior_type == 'laplace':
            intercept = numpyro.sample('intercept', 
                                      dist.Laplace(self.prior_params['mu'], 
                                                  self.prior_params['scale']))
            beta = numpyro.sample('beta', 
                                 dist.Laplace(self.prior_params['mu'], 
                                            self.prior_params['scale']).expand([n_features]))
        
        elif self.prior_type == 'cauchy':
            intercept = numpyro.sample('intercept', 
                                      dist.Cauchy(self.prior_params['loc'], 
                                                 self.prior_params['scale']))
            beta = numpyro.sample('beta', 
                                 dist.Cauchy(self.prior_params['loc'], 
                                           self.prior_params['scale']).expand([n_features]))
        
        # Linear combination: β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ
        logits = intercept + jnp.dot(X, beta)
        
        # Likelihood: Bernoulli with logistic link
        numpyro.sample('y', dist.Bernoulli(logits=logits), obs=y)
    
    def fit(self, X_train, y_train, n_samples=20000, n_burn=2000, n_chains=3):
        """
        Fit Bayesian Logistic Regression model using MCMC
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training features
        y_train : array-like, shape (n_samples,)
            Training labels (0 or 1)
        n_samples : int, default=20000
            Number of MCMC iterations
        n_burn : int, default=2000
            Number of burn-in samples
        n_chains : int, default=3
            Number of independent MCMC chains
        """
        # Standardize features (optional)
        if self.standardize and self.scaler is not None:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train
            # Calculate mean and std for potential manual scaling info
            self.feature_means = np.mean(X_train, axis=0)
            self.feature_stds = np.std(X_train, axis=0)
        
        # Convert to JAX arrays
        X_jax = jnp.array(X_train_scaled)
        y_jax = jnp.array(y_train)
        
        print(f"\n{'='*60}")
        print(f"Fitting Bayesian Logistic Regression with {self.prior_type.upper()} prior")
        print(f"MCMC Settings: {n_samples} iterations, {n_burn} burn-in, {n_chains} chains")
        print(f"Prior parameters: {self.prior_params}")
        print(f"Standardization: {'Enabled' if self.standardize else 'Disabled'}")
        print(f"{'='*60}\n")
        
        # Set up NUTS sampler with better settings
        nuts_kernel = NUTS(
            self._model,
            target_accept_prob=0.9,  # Higher acceptance rate for better sampling
            max_tree_depth=10        # Allow deeper trees
        )
        
        # Run MCMC
        self.mcmc = MCMC(
            nuts_kernel,
            num_warmup=n_burn,
            num_samples=n_samples,
            num_chains=n_chains,
            progress_bar=True,
            chain_method='parallel'  # Parallel chains for speed
        )
        
        rng_key = random.PRNGKey(42)
        self.mcmc.run(rng_key, X_jax, y_jax)
        
        # Get samples
        self.samples = self.mcmc.get_samples()
        
        print("\n✓ Model fitting completed successfully!")
        
        # Check for convergence issues
        divergences = self.mcmc.get_extra_fields()['diverging'].sum()
        if divergences > 0:
            print(f"⚠️  Warning: {divergences} divergent transitions detected!")
            print(f"   Consider: longer warmup, higher target_accept_prob")
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict probability of positive class
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns:
        --------
        probabilities : array, shape (n_samples,)
            Predicted probabilities for positive class
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get posterior means
        beta_mean = jnp.mean(self.samples['beta'], axis=0)
        intercept_mean = jnp.mean(self.samples['intercept'])
        
        # Calculate logits and probabilities
        logits = intercept_mean + jnp.dot(X_test_scaled, beta_mean)
        probabilities = 1 / (1 + jnp.exp(-logits))
        
        # Diagnostic: Print probability distribution
        probs_array = np.array(probabilities)
        print(f"\n📊 Probability Distribution Diagnostics:")
        print(f"  Min probability:    {probs_array.min():.4f}")
        print(f"  Max probability:    {probs_array.max():.4f}")
        print(f"  Mean probability:   {probs_array.mean():.4f}")
        print(f"  Std probability:    {probs_array.std():.4f}")
        print(f"  Median probability: {np.median(probs_array):.4f}")
        print(f"  Range: [{probs_array.min():.4f}, {probs_array.max():.4f}]")
        
        # Check if probabilities are too narrow
        prob_range = probs_array.max() - probs_array.min()
        if prob_range < 0.3:
            print(f"  ⚠️  WARNING: Probability range is only {prob_range:.4f}")
            print(f"      This indicates weak predictive power!")
            print(f"      Model cannot confidently distinguish between classes.")
        
        return probs_array
    
    def predict(self, X_test, threshold=0.5):
        """
        Predict class labels
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
        threshold : float, default=0.5
            Decision threshold for classification
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X_test)
        predictions = (probabilities >= threshold).astype(int)
        return predictions
    
    def find_optimal_threshold(self, X_val, y_val, metric='balanced', balance_tolerance=0.05):
        """
        Find optimal threshold to balance precision and recall
        
        Parameters:
        -----------
        X_val : array-like
            Validation features
        y_val : array-like
            Validation labels
        metric : str, {'f1', 'balanced', 'precision_recall_balance'}
            Optimization metric
        balance_tolerance : float, default=0.05
            Maximum acceptable difference between precision and recall (for balanced mode)
            
        Returns:
        --------
        optimal_threshold : float
            Threshold that optimizes the metric
        """
        print("\n" + "="*60)
        print("THRESHOLD OPTIMIZATION")
        print("="*60)
        
        probabilities = self.predict_proba(X_val)
        
        # Analyze probability distribution
        print(f"\n🔍 Analyzing {len(probabilities)} validation predictions...")
        print(f"   Class distribution in validation: {np.bincount(y_val)}")
        
        # Check if probabilities are problematic
        prob_range = probabilities.max() - probabilities.min()
        if prob_range < 0.1:
            print(f"\n❌ CRITICAL ISSUE DETECTED!")
            print(f"   Probability range is only {prob_range:.4f}")
            print(f"   This means the model has VERY WEAK discriminative power.")
            print(f"   All predictions are essentially the same!")
            print(f"\n   Possible causes:")
            print(f"   1. Features have no predictive power")
            print(f"   2. Data is random/synthetic")
            print(f"   3. Need feature engineering")
            print(f"   4. Class imbalance is too severe")
        
        # More granular threshold search
        thresholds = np.linspace(probabilities.min() - 0.01, probabilities.max() + 0.01, 500)
        
        best_score = -np.inf
        optimal_threshold = 0.5
        
        scores = []
        precisions = []
        recalls = []
        f1_scores = []
        balance_diffs = []
        accuracies = []
        
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            
            # Skip if all predictions are the same
            if len(np.unique(predictions)) < 2:
                scores.append(0)
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
                balance_diffs.append(1)
                accuracies.append(0)
                continue
            
            # Calculate confusion matrix
            try:
                tn, fp, fn, tp = confusion_matrix(y_val, predictions, labels=[0, 1]).ravel()
            except ValueError:
                if predictions.sum() == 0:
                    tp, fp = 0, 0
                    fn = y_val.sum()
                    tn = len(y_val) - fn
                else:
                    tp = y_val.sum()
                    fn = 0
                    fp = len(y_val) - tp
                    tn = 0
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
            
            # Calculate balance difference
            balance_diff = abs(precision - recall)
            balance_diffs.append(balance_diff)
            
            # Different scoring strategies
            if metric == 'f1':
                score = f1
            elif metric == 'balanced':
                score = (precision + recall) / 2
            elif metric == 'precision_recall_balance':
                if balance_diff <= balance_tolerance:
                    score = f1
                else:
                    score = f1 - (balance_diff * 2)
            
            scores.append(score)
            
            if score > best_score and precision > 0 and recall > 0:
                best_score = score
                optimal_threshold = threshold
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                best_accuracy = accuracy
        
        # Plot threshold optimization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Threshold vs Score
        axes[0, 0].plot(thresholds, scores, linewidth=2, label=f'{metric.upper()} Score', color='purple')
        axes[0, 0].axvline(optimal_threshold, color='r', linestyle='--', linewidth=2,
                   label=f'Optimal: {optimal_threshold:.3f}')
        axes[0, 0].set_xlabel('Threshold', fontsize=12)
        axes[0, 0].set_ylabel(f'{metric.upper()} Score', fontsize=12)
        axes[0, 0].set_title(f'Threshold Optimization ({self.prior_type.upper()} Prior)', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Precision-Recall vs Threshold
        axes[0, 1].plot(thresholds, precisions, linewidth=2, label='Precision', color='blue')
        axes[0, 1].plot(thresholds, recalls, linewidth=2, label='Recall', color='green')
        axes[0, 1].axvline(optimal_threshold, color='r', linestyle='--', linewidth=2,
                   label=f'Optimal: {optimal_threshold:.3f}')
        axes[0, 1].axhline(best_precision, color='blue', linestyle=':', alpha=0.5)
        axes[0, 1].axhline(best_recall, color='green', linestyle=':', alpha=0.5)
        axes[0, 1].set_xlabel('Threshold', fontsize=12)
        axes[0, 1].set_ylabel('Score', fontsize=12)
        axes[0, 1].set_title('Precision-Recall vs Threshold', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: F1 Score vs Threshold
        axes[1, 0].plot(thresholds, f1_scores, linewidth=2, label='F1 Score', color='orange')
        axes[1, 0].axvline(optimal_threshold, color='r', linestyle='--', linewidth=2,
                   label=f'Optimal: {optimal_threshold:.3f}')
        axes[1, 0].axhline(best_f1, color='orange', linestyle=':', alpha=0.5,
                   label=f'Best F1: {best_f1:.3f}')
        axes[1, 0].set_xlabel('Threshold', fontsize=12)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].set_title('F1 Score vs Threshold', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Balance Difference (|Precision - Recall|)
        axes[1, 1].plot(thresholds, balance_diffs, linewidth=2, label='|Precision - Recall|', color='red')
        axes[1, 1].axvline(optimal_threshold, color='r', linestyle='--', linewidth=2,
                   label=f'Optimal: {optimal_threshold:.3f}')
        axes[1, 1].axhline(abs(best_precision - best_recall), color='red', linestyle=':', alpha=0.5,
                   label=f'Balance Diff: {abs(best_precision - best_recall):.3f}')
        axes[1, 1].set_xlabel('Threshold', fontsize=12)
        axes[1, 1].set_ylabel('Balance Difference', fontsize=12)
        axes[1, 1].set_title('Precision-Recall Balance', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Comprehensive Threshold Analysis ({self.prior_type.upper()} Prior)', 
                    fontsize=16, y=1.00)
        plt.tight_layout()
        plt.savefig(f'threshold_optimization_{self.prior_type}.png', dpi=300)
        plt.close()
        
        print(f"\n✓ Optimal threshold: {optimal_threshold:.3f}")
        print(f"  Best {metric} score: {best_score:.4f}")
        print(f"  Precision: {best_precision:.4f} ({best_precision*100:.2f}%)")
        print(f"  Recall:    {best_recall:.4f} ({best_recall*100:.2f}%)")
        print(f"  F1 Score:  {best_f1:.4f} ({best_f1*100:.2f}%)")
        print(f"  Balance difference: {abs(best_precision - best_recall):.4f}")
        
        return optimal_threshold
    
    def plot_posterior_distributions(self):
        """Plot posterior distributions of model parameters"""
        n_params = min(6, len(self.samples['beta'][0]) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot intercept
        intercept_samples = np.array(self.samples['intercept'])
        axes[0].hist(intercept_samples, bins=50, density=True, alpha=0.7, color='steelblue')
        axes[0].axvline(np.mean(intercept_samples), color='r', linestyle='--', linewidth=2, label='Mean')
        axes[0].set_title('Intercept (β₀)', fontsize=12)
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot beta coefficients
        beta_samples = np.array(self.samples['beta'])
        for i in range(min(5, beta_samples.shape[1])):
            axes[i+1].hist(beta_samples[:, i], bins=50, density=True, alpha=0.7, color='steelblue')
            axes[i+1].axvline(np.mean(beta_samples[:, i]), color='r', linestyle='--', 
                            linewidth=2, label='Mean')
            axes[i+1].set_title(f'β{i+1}', fontsize=12)
            axes[i+1].set_xlabel('Value')
            axes[i+1].set_ylabel('Density')
            axes[i+1].legend()
            axes[i+1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Posterior Distributions ({self.prior_type.upper()} Prior)', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'posterior_distributions_{self.prior_type}.png', dpi=300)
        plt.close()
        print(f"✓ Posterior distributions saved as 'posterior_distributions_{self.prior_type}.png'")
    
    def plot_trace(self):
        """Plot MCMC trace plots for convergence diagnostics"""
        n_params = min(6, len(self.samples['beta'][0]) + 1)
        fig, axes = plt.subplots(n_params, 2, figsize=(15, 3*n_params))
        
        # Intercept traces
        intercept_samples = np.array(self.samples['intercept'])
        axes[0, 0].plot(intercept_samples, alpha=0.7)
        axes[0, 0].set_title('Trace: Intercept (β₀)')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(intercept_samples, bins=50, alpha=0.7, color='steelblue')
        axes[0, 1].set_title('Distribution: Intercept (β₀)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Beta traces
        beta_samples = np.array(self.samples['beta'])
        for i in range(min(5, beta_samples.shape[1])):
            axes[i+1, 0].plot(beta_samples[:, i], alpha=0.7)
            axes[i+1, 0].set_title(f'Trace: β{i+1}')
            axes[i+1, 0].set_ylabel('Value')
            axes[i+1, 0].grid(True, alpha=0.3)
            
            axes[i+1, 1].hist(beta_samples[:, i], bins=50, alpha=0.7, color='steelblue')
            axes[i+1, 1].set_title(f'Distribution: β{i+1}')
            axes[i+1, 1].set_ylabel('Frequency')
            axes[i+1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'MCMC Trace Plots ({self.prior_type.upper()} Prior)', 
                    fontsize=16, y=1.00)
        plt.tight_layout()
        plt.savefig(f'trace_plots_{self.prior_type}.png', dpi=300)
        plt.close()
        print(f"✓ Trace plots saved as 'trace_plots_{self.prior_type}.png'")
    
    def summary(self):
        """Print summary statistics of posterior distributions"""
        print(f"\n{'='*60}")
        print(f"Posterior Summary Statistics ({self.prior_type.upper()} Prior)")
        print(f"{'='*60}\n")
        
        # Intercept
        intercept_samples = np.array(self.samples['intercept'])
        print(f"Intercept (β₀):")
        print(f"  Mean: {np.mean(intercept_samples):.4f}")
        print(f"  Std:  {np.std(intercept_samples):.4f}")
        print(f"  95% HDI: [{np.percentile(intercept_samples, 2.5):.4f}, {np.percentile(intercept_samples, 97.5):.4f}]")
        print()
        
        # Beta coefficients
        beta_samples = np.array(self.samples['beta'])
        for i in range(beta_samples.shape[1]):
            print(f"β{i+1}:")
            print(f"  Mean: {np.mean(beta_samples[:, i]):.4f}")
            print(f"  Std:  {np.std(beta_samples[:, i]):.4f}")
            print(f"  95% HDI: [{np.percentile(beta_samples[:, i], 2.5):.4f}, {np.percentile(beta_samples[:, i], 97.5):.4f}]")
            print()
        
        # Print MCMC diagnostics
        self.mcmc.print_summary()


def load_and_preprocess_data(filepath='Medicaldataset (1).csv'):
    """
    Load and preprocess medical dataset
    """
    print(f"\n{'='*60}")
    print("Loading and Preprocessing Data")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Identify target column (likely 'Result' with 'positive'/'negative')
    target_cols = [col for col in df.columns if 'result' in col.lower() or 
                   'outcome' in col.lower() or 'target' in col.lower() or
                   'class' in col.lower()]
    
    if not target_cols:
        target_col = df.columns[-1]
    else:
        target_col = target_cols[0]
    
    print(f"✓ Target column identified: '{target_col}'")
    
    # Convert target to binary (0/1)
    if df[target_col].dtype == 'object':
        # Handle 'positive'/'negative' or similar text labels
        unique_values = df[target_col].unique()
        print(f"  Original target values: {unique_values}")
        
        # Map to 0/1 (positive=1, negative=0)
        label_map = {}
        for val in unique_values:
            val_lower = str(val).lower().strip()
            if 'pos' in val_lower or val_lower == '1':
                label_map[val] = 1
            elif 'neg' in val_lower or val_lower == '0':
                label_map[val] = 0
            else:
                # Default: first unique value = 0, second = 1
                label_map[val] = list(unique_values).index(val)
        
        df[target_col] = df[target_col].map(label_map)
        print(f"  Mapped to: {label_map}")
    
    y = df[target_col].values
    
    # Separate features and target
    exclude_cols = [target_col]
    if 'Patient_ID' in df.columns:
        exclude_cols.append('Patient_ID')
    if 'Patient ID' in df.columns:
        exclude_cols.append('Patient ID')
    if 'ID' in df.columns:
        exclude_cols.append('ID')
        
    X = df.drop(columns=exclude_cols, errors='ignore')
    
    # Handle categorical variables
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"  Converting categorical column: {col}")
            X[col] = pd.Categorical(X[col]).codes
    
    feature_names = X.columns.tolist()
    X = X.values
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n✓ Class distribution:")
    for label, count in zip(unique, counts):
        label_name = "Negative" if label == 0 else "Positive"
        print(f"   {label_name} ({label}): {count} ({count/len(y)*100:.2f}%)")
    
    print(f"\n✓ Features ({len(feature_names)}): {feature_names}")
    
    # Check for missing values
    if np.isnan(X).any():
        print(f"\n⚠️  Warning: Dataset contains {np.isnan(X).sum()} missing values")
        print(f"   Filling with median values...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    return X, y, feature_names


def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Comprehensive model evaluation"""
    print(f"\n{'='*60}")
    print(f"Evaluation Metrics: {model_name}")
    print(f"{'='*60}\n")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Detailed classification report
    print(f"\n{'-'*60}")
    print("Detailed Classification Report:")
    print(f"{'-'*60}")
    print(classification_report(y_true, y_pred, target_names=['No Risk', 'Risk'], zero_division=0))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Risk', 'Risk'],
                yticklabels=['No Risk', 'Risk'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.close()
    print(f"\n✓ Confusion matrix saved")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def main():
    """Main execution pipeline"""
    
    # 1. Load and preprocess data
    X, y, feature_names = load_and_preprocess_data('Medicaldataset (1).csv')
    
    # 2. Split data: 80% train, 10% validation, 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
    )
    
    print(f"\n✓ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 3. Train models with different priors (reduced iterations for faster testing)
    priors = {
        'gaussian': {'mu': 0.0, 'sigma': 10.0},
        'laplace': {'mu': 0.0, 'scale': 1.0},
        'cauchy': {'loc': 0.0, 'scale': 0.01}
    }
    
    results = {}
    
    for prior_name, prior_params in priors.items():
        print(f"\n\n{'#'*60}")
        print(f"Training Model with {prior_name.upper()} Prior")
        print(f"{'#'*60}")
        
        # Initialize and fit model
        blr = BayesianLogisticRegression(prior_type=prior_name, prior_params=prior_params)
        blr.fit(X_train, y_train, n_samples=3000, n_burn=500, n_chains=2)  # Faster for testing
        
        # Generate diagnostics
        blr.summary()
        blr.plot_trace()
        blr.plot_posterior_distributions()
        
        # Find optimal threshold using BALANCED precision-recall approach
        print("\n" + "="*60)
        print("Finding Optimal Threshold (Balanced Precision-Recall)")
        print("="*60)
        optimal_threshold = blr.find_optimal_threshold(X_val, y_val, 
                                                       metric='precision_recall_balance',
                                                       balance_tolerance=0.10)  # Slightly relaxed tolerance
        
        # Predict on test set with optimal threshold
        print("\n" + "="*60)
        print("Testing on Holdout Test Set")
        print("="*60)
        y_pred = blr.predict(X_test, threshold=optimal_threshold)
        y_proba = blr.predict_proba(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_proba, 
                                f'{prior_name.upper()} Prior BLR')
        
        results[prior_name] = {
            'model': blr,
            'threshold': optimal_threshold,
            'metrics': metrics
        }
    
    # 4. Compare all models
    print(f"\n\n{'='*60}")
    print("Comparison of All Models")
    print(f"{'='*60}\n")
    
    comparison_df = pd.DataFrame({
        name: data['metrics'] 
        for name, data in results.items()
    }).T
    
    print(comparison_df.to_string())
    
    # Find best model
    best_model_name = comparison_df['f1'].idxmax()
    print(f"\n✓ Best performing model: {best_model_name.upper()} Prior")
    print(f"  F1 Score: {comparison_df.loc[best_model_name, 'f1']:.4f}")
    print(f"  Accuracy: {comparison_df.loc[best_model_name, 'accuracy']:.4f}")
    print(f"  Precision: {comparison_df.loc[best_model_name, 'precision']:.4f}")
    print(f"  Recall: {comparison_df.loc[best_model_name, 'recall']:.4f}")
    print(f"  Optimal Threshold: {results[best_model_name]['threshold']:.3f}")
    
    # Save comparison plot
    comparison_df.plot(kind='bar', figsize=(12, 6), rot=0)
    plt.title('Model Comparison: Different Prior Distributions', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Prior Type', fontsize=12)
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()
    print(f"\n✓ Model comparison plot saved as 'model_comparison.png'")
    
    # Feature importance analysis
    print(f"\n\n{'='*60}")
    print("Feature Importance Analysis (Best Model)")
    print(f"{'='*60}\n")
    
    best_model = results[best_model_name]['model']
    beta_mean = np.mean(best_model.samples['beta'], axis=0)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': beta_mean,
        'Abs_Coefficient': np.abs(beta_mean)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in feature_importance['Coefficient']]
    plt.barh(range(len(feature_importance)), feature_importance['Coefficient'], color=colors)
    plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.title(f'Feature Importance ({best_model_name.upper()} Prior)', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()
    print(f"\n✓ Feature importance plot saved as 'feature_importance.png'")
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}\n")
    print(f"📊 Summary:")
    print(f"   - Best Model: {best_model_name.upper()} Prior")
    print(f"   - Test Accuracy: {comparison_df.loc[best_model_name, 'accuracy']*100:.2f}%")
    print(f"   - Test F1 Score: {comparison_df.loc[best_model_name, 'f1']*100:.2f}%")
    print(f"   - Most Important Features: {', '.join(feature_importance.head(3)['Feature'].values)}")
    print(f"\n✅ All visualizations saved to current directory")


if __name__ == "__main__":
    main()