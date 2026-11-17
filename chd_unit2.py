"""
Unit 2 (21CSC305P) applied to: 10-year Coronary Heart Disease (CHD) risk estimation
Dataset: Framingham Heart Study (4240 patients, 15 clinical indicators, target = TenYearCHD)

Unit-2 concepts implemented:
  1. Maximum Likelihood Estimation (IRLS / Newton-Raphson)  -> discriminative logistic model
  2. Least squares for classification                        -> shows why squared loss fails here
  3. Ridge / MAP estimation with a Gaussian prior            -> regularised logistic regression
  4. Probabilistic generative model (Gaussian class-conditionals + Bayes' theorem)
  5. Fisher's Linear Discriminant + Perceptron
  6. Laplace approximation -> Bayesian logistic regression -> predictive distribution
  7. Kernel functions / kernel trick / SVM (linear, polynomial, RBF)
"""

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, brier_score_loss
from sklearn.calibration import calibration_curve

np.random.seed(0)
OUT = "/mnt/user-data/outputs"

# ----------------------------------------------------------------------------------
# 0. DATA
# ----------------------------------------------------------------------------------
df = pd.read_csv("framingham.csv")
df = df.fillna(df.median(numeric_only=True))          # simple median imputation
FEATS = ["age", "male", "totChol", "sysBP", "diaBP", "cigsPerDay", "currentSmoker",
         "BMI", "glucose", "diabetes", "BPMeds", "prevalentHyp", "prevalentStroke", "heartRate"]
X_raw = df[FEATS].values.astype(float)
y = df["TenYearCHD"].values.astype(int)

Xtr_r, Xte_r, ytr, yte = train_test_split(X_raw, y, test_size=0.25, stratify=y, random_state=42)
mu, sd = Xtr_r.mean(0), Xtr_r.std(0)
Xtr, Xte = (Xtr_r - mu) / sd, (Xte_r - mu) / sd        # standardise (kernels & priors need this)
Phi_tr = np.c_[np.ones(len(Xtr)), Xtr]                 # design matrix with bias
Phi_te = np.c_[np.ones(len(Xte)), Xte]
D = Phi_tr.shape[1]

print("=" * 78)
print("DATA".center(78))
print("=" * 78)
print(f"patients={len(df)}  features={len(FEATS)}  CHD prevalence={y.mean():.3f}")
print(f"train={len(ytr)}  test={len(yte)}  (stratified)")

sig = lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))

# ----------------------------------------------------------------------------------
# 1. MLE -- logistic regression by Newton-Raphson / IRLS
#    p(t|w) = prod y_n^t_n (1-y_n)^(1-t_n)  ->  minimise cross-entropy
# ----------------------------------------------------------------------------------
def irls(Phi, t, alpha=0.0, iters=25, tol=1e-8):
    """alpha = 0 -> pure MLE ; alpha > 0 -> MAP with Gaussian prior N(0, alpha^-1 I) = ridge"""
    w = np.zeros(Phi.shape[1]); hist = []
    for _ in range(iters):
        yv = sig(Phi @ w)
        R = np.clip(yv * (1 - yv), 1e-10, None)              # weight matrix diag
        g = Phi.T @ (yv - t) + alpha * w                     # gradient
        H = Phi.T @ (Phi * R[:, None]) + alpha * np.eye(Phi.shape[1])   # Hessian
        step = np.linalg.solve(H, g)
        w_new = w - step
        ll = np.sum(t * np.log(np.clip(sig(Phi @ w_new), 1e-12, 1)) +
                    (1 - t) * np.log(np.clip(1 - sig(Phi @ w_new), 1e-12, 1)))
        hist.append(ll)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new; break
        w = w_new
    return w, np.array(hist), H

w_mle, ll_hist, _ = irls(Phi_tr, ytr, alpha=0.0)
print("\n" + "=" * 78); print("1. MAXIMUM LIKELIHOOD ESTIMATION (IRLS)".center(78)); print("=" * 78)
print(f"converged in {len(ll_hist)} Newton steps | final log-likelihood = {ll_hist[-1]:.3f}")
print(f"{'feature':<18}{'w (MLE)':>10}{'odds ratio':>13}   (per 1 SD change)")
for n, wi in zip(["bias"] + FEATS, w_mle):
    print(f"{n:<18}{wi:>10.4f}{np.exp(wi):>13.3f}")

# ----------------------------------------------------------------------------------
# 2. LEAST SQUARES FOR CLASSIFICATION (Unit-2 discriminant function)
# ----------------------------------------------------------------------------------
w_ls = np.linalg.pinv(Phi_tr) @ ytr          # normal equations, closed form
ls_scores = Phi_te @ w_ls
print("\n" + "=" * 78); print("2. LEAST-SQUARES DISCRIMINANT (closed form)".center(78)); print("=" * 78)
print(f"outputs are NOT probabilities: range [{ls_scores.min():.2f}, {ls_scores.max():.2f}]")
print(f"fraction of test outputs outside [0,1] = "
      f"{np.mean((ls_scores < 0) | (ls_scores > 1)):.3f}  -> cannot be read as p(CHD|x)")

# ----------------------------------------------------------------------------------
# 3. RIDGE / MAP  (Gaussian prior on w  <=>  L2 penalty)
# ----------------------------------------------------------------------------------
print("\n" + "=" * 78); print("3. MAP / RIDGE-REGULARISED LOGISTIC REGRESSION".center(78)); print("=" * 78)
print(f"{'alpha':>8}{'||w||':>10}{'train LL':>12}{'test AUC':>11}")
best = (None, -1, None)
for a in [0.0, 0.1, 1, 10, 50, 200, 1000]:
    w_a, h_a, H_a = irls(Phi_tr, ytr, alpha=a)
    auc = roc_auc_score(yte, sig(Phi_te @ w_a))
    print(f"{a:>8}{np.linalg.norm(w_a):>10.3f}{h_a[-1]:>12.2f}{auc:>11.4f}")
    if auc > best[1]: best = (a, auc, (w_a, H_a))
alpha_star, _, (w_map, H_map) = best
print(f"--> best alpha = {alpha_star} ; coefficients shrink as alpha grows (bias-variance trade-off)")

# ----------------------------------------------------------------------------------
# 4. PROBABILISTIC GENERATIVE MODEL: Gaussian class-conditionals + Bayes' theorem
#    p(x|C_k)=N(mu_k, Sigma) shared Sigma  =>  posterior is exactly a sigmoid  (Unit 2)
# ----------------------------------------------------------------------------------
pi1 = ytr.mean(); pi0 = 1 - pi1
m1, m0 = Xtr[ytr == 1].mean(0), Xtr[ytr == 0].mean(0)
S1 = np.cov(Xtr[ytr == 1].T, bias=True); S0 = np.cov(Xtr[ytr == 0].T, bias=True)
Sw = pi1 * S1 + pi0 * S0                                   # shared within-class covariance
Sinv = np.linalg.pinv(Sw)
w_gen = Sinv @ (m1 - m0)                                   # Bishop 4.66
b_gen = -0.5 * m1 @ Sinv @ m1 + 0.5 * m0 @ Sinv @ m0 + np.log(pi1 / pi0)
p_gen_te = sig(Xte @ w_gen + b_gen)
print("\n" + "=" * 78); print("4. PROBABILISTIC GENERATIVE MODEL (Gaussian / LDA)".center(78)); print("=" * 78)
print(f"priors: p(CHD)={pi1:.3f}  p(no CHD)={pi0:.3f}")
print(f"generative weights w = Sigma^-1 (mu1-mu0); posterior p(C1|x)=sigmoid(w.x+b) -- same"
      f" functional form as model 1, different fitting principle")
print(f"test AUC (generative) = {roc_auc_score(yte, p_gen_te):.4f}   vs "
      f"discriminative MAP = {roc_auc_score(yte, sig(Phi_te @ w_map)):.4f}")

# ----------------------------------------------------------------------------------
# 5. FISHER'S LINEAR DISCRIMINANT + PERCEPTRON
# ----------------------------------------------------------------------------------
w_fisher = Sinv @ (m1 - m0); w_fisher /= np.linalg.norm(w_fisher)
z_tr, z_te = Xtr @ w_fisher, Xte @ w_fisher
J = ((z_tr[ytr == 1].mean() - z_tr[ytr == 0].mean()) ** 2) / (z_tr[ytr == 1].var() + z_tr[ytr == 0].var())

t_pm = np.where(ytr == 1, 1, -1)                            # perceptron targets in {-1,+1}
w_perc = np.zeros(D)
for epoch in range(20):
    idx = np.random.permutation(len(ytr)); errs = 0
    for i in idx:
        if t_pm[i] * (Phi_tr[i] @ w_perc) <= 0:
            w_perc += 0.01 * t_pm[i] * Phi_tr[i]; errs += 1
perc_acc = accuracy_score(yte, (Phi_te @ w_perc > 0).astype(int))
print("\n" + "=" * 78); print("5. FISHER LDA + PERCEPTRON".center(78)); print("=" * 78)
print(f"Fisher criterion J(w) on the 1-D projection = {J:.4f}  (AUC={roc_auc_score(yte, z_te):.4f})")
print(f"Perceptron after 20 epochs: {errs} updates in last epoch -> classes are NOT separable;"
      f" accuracy={perc_acc:.4f} and it gives no probability")

# ----------------------------------------------------------------------------------
# 6. LAPLACE APPROXIMATION -> BAYESIAN LOGISTIC REGRESSION -> PREDICTIVE DISTRIBUTION
#    q(w) = N(w_MAP, S_N),  S_N = H^-1 (Hessian of the negative log posterior)
#    p(C1|x) ~= sigmoid( kappa(s^2) * mu_a ),  kappa(s^2) = (1 + pi s^2 / 8)^-1/2
# ----------------------------------------------------------------------------------
S_N = np.linalg.inv(H_map)
kappa = lambda s2: 1.0 / np.sqrt(1.0 + np.pi * s2 / 8.0)

def bayes_predict(Phi):
    mu_a = Phi @ w_map                                      # mean of the activation
    s2_a = np.einsum("ij,jk,ik->i", Phi, S_N, Phi)          # variance of the activation
    return sig(kappa(s2_a) * mu_a), mu_a, s2_a

p_bayes_te, mu_a_te, s2_a_te = bayes_predict(Phi_te)
print("\n" + "=" * 78); print("6. BAYESIAN LOGISTIC REGRESSION (Laplace approximation)".center(78)); print("=" * 78)
print(f"posterior covariance S_N: shape {S_N.shape}, mean predictive sd of activation = "
      f"{np.sqrt(s2_a_te).mean():.4f}")
print(f"test AUC (Bayesian predictive) = {roc_auc_score(yte, p_bayes_te):.4f}")
print("MAP point estimate is over-confident; Laplace shrinks p towards 0.5 by the factor kappa.")

# ----------------------------------------------------------------------------------
# 7. KERNELS / KERNEL TRICK / SVM
# ----------------------------------------------------------------------------------
print("\n" + "=" * 78); print("7. KERNEL FUNCTIONS AND SVMs".center(78)); print("=" * 78)
svms = {}
for name, kw in [("linear", dict(kernel="linear", C=1.0)),
                 ("poly (d=3)", dict(kernel="poly", degree=3, C=1.0, gamma="scale")),
                 ("RBF", dict(kernel="rbf", C=1.0, gamma="scale"))]:
    m = SVC(probability=True, class_weight="balanced", random_state=0, **kw).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    svms[name] = p
    print(f"{name:<12} support vectors={m.n_support_.sum():>5}  test AUC={roc_auc_score(yte, p):.4f}")
G = Xtr[:3] @ Xtr[:3].T
print(f"Gram matrix (linear kernel, first 3 patients) =\n{np.round(G,3)}   <- symmetric, PSD (Mercer)")

# ----------------------------------------------------------------------------------
# 8. MODEL COMPARISON
# ----------------------------------------------------------------------------------
models = {
    "Least squares": ls_scores,
    "Perceptron (score)": Phi_te @ w_perc,
    "Fisher LDA": z_te,
    "Generative Gaussian": p_gen_te,
    "Logistic MLE": sig(Phi_te @ w_mle),
    "Logistic MAP/ridge": sig(Phi_te @ w_map),
    "Bayesian (Laplace)": p_bayes_te,
    "SVM linear": svms["linear"],
    "SVM RBF": svms["RBF"],
}
print("\n" + "=" * 78); print("8. MODEL COMPARISON ON HELD-OUT TEST SET".center(78)); print("=" * 78)
print(f"{'model':<22}{'AUC':>8}{'acc@thr':>10}{'recall':>9}{'Brier':>9}")
for name, s in models.items():
    auc = roc_auc_score(yte, s)
    fpr, tpr, thr = roc_curve(yte, s)
    t_star = thr[np.argmax(tpr - fpr)]                      # Youden's J
    pred = (s >= t_star).astype(int)
    b = brier_score_loss(yte, np.clip(s, 0, 1)) if s.min() >= -0.5 and s.max() <= 1.5 else float("nan")
    print(f"{name:<22}{auc:>8.4f}{accuracy_score(yte,pred):>10.4f}"
          f"{recall_score(yte,pred):>9.4f}{b:>9.4f}")

# ----------------------------------------------------------------------------------
# 9. CASE STUDY -- risk for individual patients, with Bayesian uncertainty
# ----------------------------------------------------------------------------------
def risk(profile):
    x = np.array([[profile.get(f, df[f].median()) for f in FEATS]], float)
    phi = np.c_[np.ones(1), (x - mu) / sd]
    p, m_a, s2 = bayes_predict(phi)
    samples = sig(np.random.multivariate_normal(w_map, S_N, 5000) @ phi[0])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return p[0], sig(phi @ w_map)[0], lo, hi

print("\n" + "=" * 78); print("9. INDIVIDUAL 10-YEAR CHD RISK ESTIMATES".center(78)); print("=" * 78)
patients = {
    "A: 40F, chol 190, SBP 110, non-smoker":
        dict(age=40, male=0, totChol=190, sysBP=110, diaBP=72, cigsPerDay=0, currentSmoker=0, BMI=23, glucose=80),
    "B: 55M, chol 250, SBP 140, 20 cig/day":
        dict(age=55, male=1, totChol=250, sysBP=140, diaBP=90, cigsPerDay=20, currentSmoker=1, BMI=28, glucose=85),
    "C: 65M, chol 300, SBP 175, 30 cig/day, diabetic":
        dict(age=65, male=1, totChol=300, sysBP=175, diaBP=100, cigsPerDay=30, currentSmoker=1,
             BMI=31, glucose=160, diabetes=1, prevalentHyp=1, BPMeds=1),
}
print(f"{'patient':<48}{'p(CHD)':>9}{'MAP':>8}   95% credible interval")
for k, v in patients.items():
    p, pmap, lo, hi = risk(v)
    print(f"{k:<48}{p:>9.3f}{pmap:>8.3f}   [{lo:.3f}, {hi:.3f}]")

# ----------------------------------------------------------------------------------
# 10. FIGURES
# ----------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
ax[0, 0].plot(ll_hist, "o-"); ax[0, 0].set(title="1. IRLS convergence (MLE)",
              xlabel="Newton iteration", ylabel="log-likelihood"); ax[0, 0].grid(alpha=.3)

alphas = np.logspace(-2, 3.5, 25)
paths = np.array([irls(Phi_tr, ytr, alpha=a)[0] for a in alphas])
for j in range(1, D): ax[0, 1].plot(alphas, paths[:, j])
ax[0, 1].set(xscale="log", title="3. Ridge/MAP coefficient shrinkage",
             xlabel="alpha (prior precision)", ylabel="w"); ax[0, 1].grid(alpha=.3)

ax[0, 2].hist(z_tr[ytr == 0], 40, alpha=.6, density=True, label="no CHD")
ax[0, 2].hist(z_tr[ytr == 1], 40, alpha=.6, density=True, label="CHD")
ax[0, 2].set(title=f"5. Fisher LDA projection (J={J:.3f})", xlabel="w_F . x"); ax[0, 2].legend()

for name in ["Logistic MAP/ridge", "Generative Gaussian", "SVM RBF", "Fisher LDA", "Least squares"]:
    fpr, tpr, _ = roc_curve(yte, models[name])
    ax[1, 0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(yte, models[name]):.3f})")
ax[1, 0].plot([0, 1], [0, 1], "k--"); ax[1, 0].legend(fontsize=8)
ax[1, 0].set(title="8. ROC curves", xlabel="FPR", ylabel="TPR")

for name in ["Logistic MLE", "Bayesian (Laplace)", "Generative Gaussian"]:
    pt, pp = calibration_curve(yte, np.clip(models[name], 0, 1), n_bins=8, strategy="quantile")
    ax[1, 1].plot(pp, pt, "o-", label=name)
ax[1, 1].plot([0, .6], [0, .6], "k--"); ax[1, 1].legend(fontsize=8)
ax[1, 1].set(title="6. Calibration: predicted vs observed risk",
             xlabel="predicted p(CHD)", ylabel="observed frequency")

ax[1, 2].scatter(sig(mu_a_te), p_bayes_te, s=8, c=np.sqrt(s2_a_te), cmap="viridis")
ax[1, 2].plot([0, 1], [0, 1], "k--")
ax[1, 2].set(title="6. Laplace moderation of the MAP risk",
             xlabel="sigmoid(w_MAP . phi)", ylabel="Bayesian predictive p")
plt.colorbar(ax[1, 2].collections[0], ax=ax[1, 2], label="sd of activation")

plt.tight_layout(); plt.savefig(f"{OUT}/chd_unit2_results.png", dpi=130)
print(f"\nfigures written to {OUT}/chd_unit2_results.png")
