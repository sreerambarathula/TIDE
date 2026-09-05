# Journal Manuscript Creation Guide & Master Blueprint

**Target Journal:** *Reliability Engineering & System Safety* (Elsevier, Impact Factor ~9.4) / *Nuclear Engineering and Design*  
**Article Title:** *Mitigation Strategies for Neural Surrogate Breakdown at Flow Boiling Instability Boundaries to Ensure Safe System Reliability*

---

## 🌐 GLOBAL MANUSCRIPT DRAFTING GUIDELINES

### 1. Academic Tone, Style & Physical Grounding
* **Authoritative Academic Prose:** Maintain a formal, objective, rigorous academic journal tone matching Elsevier / RESS standards. Eliminate all colloquialisms, informal expressions, and hand-waving.
* **Physical Mechanism Rule:** Never describe numerical results simply as "increasing" or "decreasing metrics." Ground every observation directly in its physical thermal-hydraulic or geometric mechanism (e.g., vapor density disparity, convective transport delay, liquid acceleration, mode coalescence at the double-zero root, knife-edge stability corridor).
* **Balanced Novelty Framing:** Ground the contribution in existing literature (Wang, Krishnapriyan, Echard, Sudret, Theler); frame the novelty with academic precision as the first investigation linking codimension-2 bifurcation geometry ($\delta^{1.00}$) to localized SciML metric decoupling.

### 2. Structural & Architectural Standards (Professor-Approved)
* **Streamlined IMRaD Format:** Strictly follow the 4-part classical structure: (1) Introduction, (2) Materials and Methods, (3) Results and Discussion, (4) Conclusions.
* **No Orphan / Fragmented Subsections:** Every subsection must be substantial, cohesive, and anchored by dedicated multi-panel figures or tables. Introduction and Conclusions must flow as continuous, cohesive prose without sub-subsections.
* **Nomenclature Compilation Rule:** Hold the Nomenclature section until all text and equations are finalized to ensure 100% mutual consistency.

### 3. Numerical Data Traceability & Integrity
* **Strict Empirical Traceability:** Zero ungrounded or placeholder numbers. All quantitative claims ($R^2$, RMSE, boundary error multipliers, coordinates, $W$-statistics, $p$-values) must be pulled directly from verified database files (`data/generated/`), script logs, and test suites.
* **Consistent Precision Conventions:** Maintain uniform reporting (e.g., coordinates to 3–6 decimals, $R^2$ to 3–4 decimals, RMSE to 3 decimals, $p$-values with exact thresholds $p < 0.005$).

### 4. Citation Strategy & Literature Prioritization
* **Target Volume:** 45 to 50 citations total across the manuscript (with 25–35 concentrated in Section 1).
* **Priority 1 (Mandatory RESS & Reliability Literature):** Heavily cite foundational and recent *Reliability Engineering & System Safety* papers on surrogate safety certification, active learning limit-state methods (AK-MCS, Kriging/MLP reliability by Sudret, Echard, Bect, Marelli, Zio), and digital twin validation.
* **Priority 2 (Thermal-Hydraulics & Bifurcation Benchmarks):** Anchor physics to Ishii & Zuber (1970), Clausse & Lahey (1991), Theler et al. (2010), Pandey & Singh (2017), and Hurley (2023).
* **Priority 3 (SciML Failure Modes):** Cite foundational spectral bias and optimization landscape studies (Wang et al. 2022, Krishnapriyan et al. 2021, Tancik et al. 2020, Rahaman et al. 2019).

### 5. Multi-Panel Visual Presentation Standards
* **High-Impact Visual Suites:** All 8 major figures (`[Figure 1]` to `[Figure 8]`) must be comprehensive, publication-grade multi-panel figures (4 to 6 connected subpanels labeled (a), (b), (c)...) loaded with real physics, continuous contour fields, spectra, and statistical distributions.
* **Cohesive Narrative Flow:** Subpanels within each figure must connect logically (e.g., physical domain $\to$ state evolution $\to$ dynamic trajectories $\to$ bifurcation manifolds), avoiding isolated or redundant individual plots.
* **Self-Contained Captions:** Every figure and table must include a comprehensive caption detailing axis units, curve definitions, symbol legends, and the core engineering takeaway.

### 6. Mathematical Rigor & Code Correspondence
* **Standard LaTeX Syntax:** Format all equations in standard LaTeX notation for seamless KaTeX/MathJax rendering.
* **Explicit Definition:** Every variable, subscript, and parameter must be explicitly defined immediately following its first appearance in an equation.
* **1-to-1 Codebase Mapping:** Ensure all analytical formulas maintain exact, verified equivalence with their source implementations in `src/tide/`.

---

## Master Manuscript Skeleton (Streamlined IMRaD Format)

```text
NOMENCLATURE

1. INTRODUCTION

2. MATERIALS AND METHODS
   2.1 Thermal-Hydraulic Modeling and Instability Formulations
       [Figure 1]
   2.2 Moving-Boundary Dynamics and Machine-Precision Continuation Algorithms
       [Figure 2]
       [Table 1]
   2.3 Machine Learning Surrogate Architectures and Statistical Validation Protocols
       [Figure 3]

3. RESULTS AND DISCUSSION
   3.1 Ground-Truth Bifurcation Topologies and Stable Window Scaling
       [Figure 4]
   3.2 The Metric Decoupling Phenomenon and Diagnostic Root-Cause Analysis
       [Figure 5]
       [Figure 6]
   3.3 Comparative Performance of Remediation Strategies and Statistical Validation
       [Figure 7]
       [Table 2]
   3.4 Reliability Certification Framework and Industrial Implications
       [Figure 8]

4. CONCLUSIONS

DATA AND CODE AVAILABILITY

REFERENCES

APPENDIX / SUPPLEMENTARY MATERIAL
   Appendix A: Complete Algebraic Derivation of the Clausse–Lahey Momentum ODE System
   Appendix B: Numerical Continuation Algorithms, Solver Verification, and Nodal Convergence
   Appendix C: Hyperparameter Specifications, Training Configurations, and Extended Statistical Tables
```

---

## 🎨 Master Multi-Panel Visual Suites Specification (Figures 1–8)

### **Figure 1: Multiscale Thermal-Hydraulic Dynamics and Genesis of Flow Instabilities (6 Connected Panels)**
* **Panel (a) Physical Zonation & Moving Phase Boundary:** Vertical boiling channel ($0 \le z \le \lambda(t) \le 1$) with axial enthalpy profile $h(z)$, sensible heating zone, boiling boundary $\lambda(t)$, two-phase mixture zone, and plenum pressure drop headers.
* **Panel (b) Phase-Change Density Disparity & Quality Paradox:** High-pressure (70 bar) density ratio ($\rho_f/\rho_g \approx 20.5$) and the non-linear void fraction curve $\alpha(x)$ highlighting the $5\%$ quality $\to 51.3\%$ void fraction inflection.
* **Panel (c) Internal S-Curve Force Decomposition:** Breakdown of the 3 competing internal pressure drop components ($\Delta P_{fric} \propto u^2$, $\Delta P_{accel}$, $\Delta P_{grav}$) generating the negative-slope regime $\partial \Delta P / \partial G < 0$.
* **Panel (d) Static Ledinegg Excursion & Dynamic Flow Jump:** Internal S-characteristic intersected by external pump curves, illustrating the hysteresis loop and the runaway catastrophic jump $B \to C$ and $B \to A$.
* **Panel (e) Dynamic Density-Wave Transit Delay:** Spatiotemporal waterfall map $z(t)$ showing enthalpy perturbation packets propagating at speed $u(z)$ with transit delay $\tau = \int_{\lambda}^1 dz/u(z)$, generating $180^\circ$ out-of-phase acoustic feedback.
* **Panel (f) Phase Portrait & Limit-Cycle Saturation:** Non-linear phase trajectories in the $(u_i, \lambda, m)$ state-space showing the Hopf limit-cycle attractor and time-domain velocity oscillations $u_i(t)$.

---

### **Figure 2: Moving-Boundary ODE Dynamics, Discretization Pathology, and Machine-Precision Continuation (6 Connected Panels)**
* **Panel (a) The Odd-Node Energy Pumping Pathology:** Transient time integration trajectories comparing odd node counts ($N_1 = 1, 3$, unphysical divergence to `NaN` at $t \approx 16$) vs. even node counts ($N_1 = 2, 4, 16$).
* **Panel (b) Multi-Node Enthalpy Field Evolution:** Spatiotemporal reconstruction of the internal enthalpy fronts $l_n(t)$ transitioning cleanly into steady boiling.
* **Panel (c) Spectral Frequency Analysis (FFT):** Power spectral density of flow oscillations isolating the fundamental DWO acoustic resonance frequency ($\omega \approx 1.83\text{ rad/s}$) and higher harmonics.
* **Panel (d) Mesh Independence & Machine-Precision Consistency:** Relative error $< 10^{-15}$ across node counts ($N_1 = 2 \dots 16$) matching the analytical closed-form Euler relation Eq. (26).
* **Panel (e) Automatic Differentiation Fold Continuation:** The $\partial Eu / \partial N_{pch} = 0$ bisection landscape computed via `jax.grad` to machine tolerance ($10^{-12}$).
* **Panel (f) 2D Newton Sum-and-Product Solver Convergence:** Log-scale residual trajectory $\|(R_1, R_2)\| \to 10^{-12}$ of the polynomial trace/determinant solver in `double_zero.py`, contrasting against the divergent raw-eigenvalue Newton solver.

---

### **Figure 3: Global Bifurcation Maps and the Geometry of the "Knife-Edge" Stability Corridor (4 Connected Panels)**
* **Panel (a) Global Operational Map $(N_{sub}, N_{pch})$:** Full operating domain showing the lower Fold manifold (Ledinegg boundary) and upper Hopf manifold (DWO boundary) across parameter space, locating Study Points A, B, and C.
* **Panel (b) Topological Contrast: Transversal Crossing (Point A) vs. Tangent Cusp (Point B):** Side-by-side high-resolution geometric comparison showing non-degenerate angle crossing ($\Delta\text{slope} \approx 0.92$) vs. tangential contact ($\Delta\text{slope} = 0$) where the stable operating window pinches to zero width at the double-zero vertex $\star$.
* **Panel (c) Multi-Decade Empirical Window Scaling ($\delta^{1.00}$):** 4-decade log-log continuation data ($\delta \in [10^{-4}, 10^0]$) showing exact linear power-law scaling $\Delta N_{pch} \propto \delta^{\mathbf{1.00}}$ ($R^2 = 0.99999985$) compared against naive quadratic tangency ($\delta^2$).
* **Panel (d) Spatial Compression of Stability Margin Profiles:** 1D transverse cross-sections of the continuous margin field $g(N_{sub}, N_{pch})$ as $\delta \to 0$, illustrating how the permissible operating margin physically collapses into a razor-thin corridor.

---

### **Figure 4: The Metric Decoupling Phenomenon: The False Illusion of Global $R^2$ (6 Connected Panels)**
* **Panel (a) Global Test Set Parity Plot:** Predicted vs. true stability margins for standard $3\times128$ MLP displaying $R^2 = 0.9994$ and tight diagonal alignment.
* **Panel (b) Spatial Sampling Density:** Space-filling Latin Hypercube Sampling (LHS) design overlaid on ground-truth margin contours $g(\mathbf{x})$.
* **Panel (c) Point A Spatial Error Heatmap (Negative Control):** Uniformly low error ($< 0.012$) across the entire transversal domain with zero boundary localized distortion.
* **Panel (d) Point B Spatial Error Heatmap (BT Tangency):** Localized error explosion ($3.5\times\text{--}10\times$ spike, $\text{RMSE} > 0.083$) concentrated exclusively along the tangent cusp.
* **Panel (e) 3D Error Surface Topography:** 3D elevation map visualizing the sharp "error mountain ridge" running precisely along the safety boundary $g = 0$.
* **Panel (f) Transverse 1D Error Slices:** Cross-sectional error profiles across the boundary comparing the flat line of Point A against the localized Gaussian-like error spike at Point B.

---

### **Figure 5: Diagnostic Investigation: Refuting Gradient Vanishing and Proving Spectral Bias (6 Connected Panels)**
* **Panel (a) Loss Gradient Vector Field $\nabla g(\mathbf{x})$:** 2D directional arrows of the true margin gradient across parameter space, confirming steep, well-behaved slopes across the boundary.
* **Panel (b) Empirical Gradient Norm Distribution $\|\nabla g\|$:** Histograms comparing near-boundary points ($\|\nabla g\| = 1.31 \pm 0.15$) vs. far-field points ($\|\nabla g\| = 1.34 \pm 0.12$), definitively refuting gradient vanishing.
* **Panel (c) 2D Spatial Fourier Power Spectrum (Ground Truth):** 2D frequency domain power $|F(k_x, k_y)|^2$ showing significant high-frequency spectral content along the acute tangent cusp.
* **Panel (d) 2D Spatial Fourier Power Spectrum (Standard MLP Reconstruction):** 2D frequency domain spectrum of the neural surrogate, visually exposing the severe low-pass attenuation of high spatial frequencies.
* **Panel (e) Spectral Reconstruction Error vs. Spatial Frequency:** Radial power spectral error curve showing exponential error growth at high spatial frequencies (the signature of Spectral Bias).
* **Panel (f) Dual-Scale Loss Trajectories:** Training convergence histories contrasting rapid global MSE decay against the stagnation of localized near-boundary error.

---

### **Figure 6: Principled Remediation Taxonomy: Feature, Input, Data, and Loss Optimizations (4 Connected Panels)**
* **Panel (a) Orthogonal Architectural Taxonomy:** Flowchart diagram detailing the 4 targeted interventions: Feature Space (Fourier Features), Input Space (Log-Distance Prior), Data Space (Adaptive Sampling), and Loss Optimization Space (Boundary-Weighted Loss).
* **Panel (b) Multi-Scale Random Fourier Features Mapping:** Visualization of the Gaussian frequency projection $\mathbf{B}\mathbf{x}$ and its multi-frequency basis functions.
* **Panel (c) Continuous Boundary Weight Field $w(\mathbf{x})$:** Heatmap and 3D surface of the inverse-distance loss weighting $w(\mathbf{x}) = 1/(|g_{true}| + \varepsilon_w)$, showing hyper-concentration at the $g=0$ frontier.
* **Panel (d) Active Learning Sample Allocation:** Spatial distribution of adaptive sampling iterations (AK-MCS style) demonstrating point clustering along high-uncertainty boundary zones.

---

### **Figure 7: Multi-Seed Statistical Benchmark and Pareto Frontier Optimization (6 Connected Panels)**
* **Panel (a) Multi-Seed Absolute Near-Boundary RMSE Boxplots:** 10-seed distributions comparing Baseline MLP, Fourier Features, Log-Distance MLP, and Boundary-Weighted Loss across Point B and Point C.
* **Panel (b) Reconstructed Safety Frontier Overlays ($g = 0$):** High-resolution contour overlays showing how Baseline MLP blurs the cusp while Boundary-Weighted Loss sharply recovers the true knife-edge vertex.
* **Panel (c) Boundary Error vs. Distance to BT ($\delta$):** Scatter plot showing error as a function of proximity to the singularity, illustrating uniform error suppression by Boundary-Weighted Loss across all scales.
* **Panel (d) The Safety-Critical Pareto Frontier:** Trade-off curve plotting Near-Boundary RMSE vs. Far-Field RMSE, proving Boundary-Weighted Loss strictly dominates the Pareto optimal front.
* **Panel (e) Empirical Cumulative Distribution Functions (ECDF):** 10-seed ECDF curves showing complete leftward error distribution shift for Boundary-Weighted Loss.
* **Panel (f) Non-Parametric Wilcoxon Paired Difference Distributions:** Boxplots of seed-by-seed paired error differences confirming $p = 0.00195 < 0.005$ statistical superiority.

---

### **Figure 8: Reliability Certification Framework, Industrial Licensing, and Multi-Physics Universality (4 Connected Panels)**
* **Panel (a) The Four Golden Rules Infographic:** Structured regulatory guidance (Prohibition of Global $R^2$, Continuation Pre-Screening, Boundary-Weighted Training, Absolute Near-Boundary Certification).
* **Panel (b) Industrial Digital Twin Licensing Decision Tree:** Step-by-step flowchart for deploying certified AI surrogates in nuclear power plants, thermal boilers, and GPU cooling loops.
* **Panel (c) Scale-Up from 1D Channels to Natural-Circulation Loops:** Comparison between the 1D single-channel testbed ($\Lambda = 0.001$) and full multi-channel natural-circulation loops with risers/downcomers ($\Lambda \approx 3\text{--}6$), illustrating invariant bifurcation topology.
* **Panel (d) Cross-Domain Multi-Physics Universality Map:** Extending the metric decoupling failure mode and boundary-weighted remediation to aeroelastic flutter boundaries in aerospace and thermal runaway boundaries in chemical engineering.

---

## Section-by-Section Detailed Planning

### 0. Nomenclature & Acronyms [HOLD UNTIL FINAL DRAFT IS COMPLETE]
* **Compilation Rule:** Compile dynamically during drafting; verify 100% mutual consistency between text, equations, and symbol tables upon final section completion.
* **Categories:** Latin Symbols (with SI units), Greek Symbols (with SI units), Dimensionless Groups, Subscripts/Superscripts, Acronyms.

---

### 1. Introduction (Comprehensive 5-Paragraph Narrative Structure)
* **Citation Allocation:** **25 to 35 citations** (out of ~50 total in the paper), with **explicit priority given to *Reliability Engineering & System Safety* (RESS)** papers.
* **Data Traceability Rule:** All quantitative performance claims ($R^2$, RMSE, boundary error multipliers, coordinates) must be extracted directly from verified database runs.

#### Paragraph 1: Industrial Motivation & The Rise of Thermal-Hydraulic Digital Twins
* *Narrative Core:* Rapid transition in nuclear power (BWR cores), industrial boilers, concentrated solar systems, and GPU microchannel cooling from heavy transient codes (RELAP5, TRACE) toward real-time physics-informed AI digital twins and surrogate models.
* *Safety Context:* The primary mandate is accurately predicting safe operational envelopes to prevent critical heat flux (CHF), dryout, and flow excursions.
* *Citations (6–8 papers):* Thermal-hydraulic digital twins and nuclear safety monitoring literature (Hurley 2023, March-Leuba, digital twins in energy systems).

#### Paragraph 2: Current AI Reliability Practice & The "Global Accuracy" Blindspot
* *Narrative Core:* How industrial engineering teams validate surrogates using aggregate dataset metrics ($R^2 > 0.99$, low global RMSE).
* *The Unverified Assumption:* The pervasive, unverified belief that high global variance explained guarantees uniform local fidelity across safety limit-state boundaries.
* *Citations (8–10 papers):* **RESS foundational and recent literature** on surrogate reliability, active learning limit-state methods, and safety margins (Sudret, Echard, Bect, Marelli, Zio, Modarres on Kriging/MLP reliability, AK-MCS, failure boundaries).

#### Paragraph 3: Physical Boiling Instabilities & The Hidden Topological Singularity
* *Narrative Core:* Physical nature of flow boiling systems with coupled static excursive collapse (Ledinegg) and dynamic convective oscillations (Density-Wave Oscillations).
* *The Geometric Singularity:* In multi-parameter space $(N_{sub}, N_{pch})$, these instability manifolds intersect at singular codimension-2 bifurcation points (Bogdanov–Takens points) where safe operating windows pinch shut.
* *Citations (6–8 papers):* Flow boiling instability fundamentals and bifurcation theory (Ishii & Zuber 1970, Clausse & Lahey 1991, Theler et al. 2010, Delmastro & Clausse 1994, Pandey & Singh 2017).

#### Paragraph 4: Research Questions, Metric Decoupling Discovery, and Core Results
* *Narrative Core:* Explicitly states the 3 guiding research questions: (1) Does high global $R^2$ guarantee safety boundary accuracy near singularities? (2) What physical mechanism causes localized neural network breakdown at tangent cusps? (3) How can surrogate training be reformulated to eliminate this vulnerability?
* *Summary of Discoveries:* Discovery of **Metric Decoupling** ($3.5\times\text{--}10\times$ boundary error spikes at BT points despite $R^2 > 0.999$; zero spike at non-degenerate crossings); refutation of gradient vanishing; proof of Spectral Bias on linear cusps ($\delta^{1.00}$); validation of the **Boundary-Weighted Loss** remediation (~57% error reduction, $p < 0.005$).
* *Citations (5–7 papers):* Physics-informed machine learning failure modes, spectral bias, and optimization landscapes (Wang et al. 2022, Krishnapriyan et al. 2021).

#### Paragraph 5: Structural Roadmap of the Manuscript
* *Narrative Core:* Guide mapping Section 2 (Materials & Methods) $\to$ Section 3 (Results & Discussion) $\to$ Section 4 (Conclusions).

---

### 2. Materials and Methods

#### 2.1 Thermal-Hydraulic Modeling, Governing Equations, and Model Selection (`[Figure 1]`)

##### Paragraph 1: Physical System Configuration & Boundary Conditions
* *Physical Setup:* Vertical heated channel (height $L$, flow area $A_c$, heated perimeter $P_h$, uniform heat flux $q''$).
* *Axial Zonation:* Subcooled liquid enters at $z = 0$ ($T_{in} < T_{sat}$); sensible heat raises liquid enthalpy to saturation ($h = h_f$) at the moving boiling boundary coordinate $z = \lambda(t)$; two-phase mixture flows across $\lambda(t) \le z \le L$ and exits at $z = L$.
* *Plenum Boundary Condition Justification:* Imposing constant channel pressure drop $\Delta P_{channel} = \Delta P_{ext}$ models the canonical multi-channel header configuration (BWR cores, steam boilers) where thousands of parallel tubes share common plenums, allowing flow rate and boiling boundary to vary freely as dynamic variables.

##### Paragraph 2: Phase-Change Thermodynamics & The 5% Quality Paradox
* *Density Disparity:* Liquid density $\rho_f \approx 740\text{ kg/m}^3$, vapor density $\rho_g \approx 36\text{ kg/m}^3$ at 70 bar ($\rho_f / \rho_g \approx 20\text{--}30\times$).
* *Quality-to-Void Formulation:* Homogeneous relation:
  $$\alpha = \frac{x}{x + (1 - x)\left(\frac{\rho_g}{\rho_f}\right)}$$
* *The 5% Mass Paradox:* Mathematical proof showing $x = 0.05 \implies \alpha = 51.3\%$ (5% steam by mass occupies $>51\%$ of channel volume).
* *Continuity & Acceleration:* Mass continuity forcing steep exit fluid acceleration: $u(z) = u_i \cdot \frac{\rho_f}{\rho(z)}$.

##### Paragraph 3: Continuous 1D Conservation Laws
* *Mass Conservation PDE:* $\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial z} = 0$
* *Momentum Conservation PDE:* $\frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2)}{\partial z} = -\frac{\partial P}{\partial z} - \rho g - \frac{f \rho u^2}{2 D_h}$
* *Energy Conservation PDE:* $\frac{\partial (\rho h)}{\partial t} + \frac{\partial (\rho u h)}{\partial z} = q'' \frac{P_h}{A_c}$

##### Paragraph 4: Model Selection Justification — The Homogeneous Equilibrium Model (HEM)
* *Physical Definition:* Homogeneous (phases move at locked velocity, Slip $S = 1.0$) and Thermal Equilibrium (both phases at $T_{sat}$).
* *The "Testbed Principle" Justification:* Why HEM is an intentional methodological strength for AI reliability research:
  1. *Machine-Precision Ground Truth:* Provides closed-form, exact steady states ($10^{-15}$) and analytical Jacobian matrices without empirical correlation noise or iteration tolerances, enabling unambiguous error attribution to the neural network.
  2. *High-Pressure Validity:* At 70 bar (BWR operating pressure), buoyancy is weak and HEM neutral stability boundaries match complex drift-flux codes within 5% to 10%.
  3. *Topological Invariance:* All qualitative bifurcation geometry (Fold curves, Hopf limit cycles, and Bogdanov–Takens tangencies) is preserved identically under HEM.

##### Paragraph 5: The Dual Instability Mechanisms (Ledinegg Excursion & DWO)
* *Static Excursive (Ledinegg):* Competition between velocity friction ($\propto u^2$, pushing $\Delta P \uparrow$) and vapor volume reduction (pushing $\Delta P \downarrow$) creates the S-shaped internal characteristic curve. The negative-slope segment ($\frac{\partial \Delta P}{\partial G} < 0$) causes middle equilibrium (Point B) to collapse in a runaway excursion to Point A or C.
* *Dynamic Density-Wave Oscillations (DWO):* Enthalpy transit delay $\tau = \int_{\lambda}^{L} \frac{dz}{u(z)}$ creates a $180^\circ$ out-of-phase exit pressure spike that feeds back to amplify inlet oscillations into non-linear limit cycles.

---

#### 2.2 Moving-Boundary Dynamics and Machine-Precision Continuation Algorithms (`[Figure 2]`, `[Table 1]`)

##### Paragraph 1: The Clausse–Lahey Moving-Boundary ODE Reduction
* *Why Moving Boundaries over Fixed Grids:* Fixed Eulerian CFD meshes suffer from artificial numerical diffusion at moving phase interfaces and require expensive iterative solvers, making analytical Jacobian evaluation and continuation impossible.
* *Leibniz Spatial Integration across $\lambda(t)$:*
  - *Boiling Boundary Kinematics ODE:* $\frac{d\lambda}{dt} = 2(u_i - \lambda)$
  - *Two-Phase Dynamic Mass Inventory ODE:* $\frac{dm}{dt} = u_i - \rho_e \cdot u_e$, with exit velocity $u_e = u_i + N_{sub}(1 - \lambda)$ and exit density $\rho_e = 1/r$ solved algebraically from $(1 - \lambda)\frac{\ln(r)}{r - 1} = m - \lambda$.
  - *Fluid Momentum Acceleration ODE:* $\frac{du_i}{dt} = -\frac{1}{m}\Big[ u_i \dot{m} + \text{Bracket} + \text{Friction} + \text{Rest} \Big]$.

##### Paragraph 2: Multi-Node Enthalpy Discretization & The Odd/Even Node Pathology
* *Multi-Node Discretization:* Discretizing single-phase liquid enthalpy profile into $N_1$ moving nodes ($l_1, \dots, l_{N_1-1}, \lambda$).
* *The Odd-Node Divergence Pathology:* Odd node counts ($N_1 = 1, 3$) suffer an unphysical numerical energy pumping pathology that diverges to `NaN` by $t \approx 16$. Even node counts ($N_1 = 2, 4, 16$) eliminate the pathology, producing smooth, bounded, physically validated limit cycles ($u_i \in [0.10, 0.85]$).

##### Paragraph 3: Dimensionless Parameterization (`[Table 1]`)
* *7 Scaling Parameters:* $N_{sub}$ (inlet coldness), $N_{pch}$ (heating power), $Fr$ (inertia vs. gravity: $v_0^2/gL$), $\Lambda$ (wall friction: $fL/2D_h$), $k_{in}$ (inlet orifice, stabilizing), $k_{out}$ (exit restriction, destabilizing), $Eu$ (dimensionless pressure drop: $\Delta P / \rho_f v_0^2$).

##### Paragraph 4: Continuation Solvers for Bifurcation Manifolds
* *Fold Manifold Continuation:* Automatic differentiation gradient bisection (`jax.grad`) on closed-form Euler relation to locate $\frac{\partial Eu}{\partial N_{pch}} = 0$ to $10^{-12}$ precision.
* *Hopf Neutral Stability Continuation:* Dynamic Jacobian construction $\mathbf{J} = \left.\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right|_{\mathbf{x}^*}$ using forward-mode automatic differentiation (`jax.jacfwd`), followed by bisection on leading complex-conjugate eigenvalue real part to find $\mathrm{Re}(\mu) = 0$ to $10^{-8}$ precision.
* *2D Newton Sum/Product Solver for BT Points (`double_zero.py`):* Solving smooth polynomial invariants $R_1 = \mathrm{Tr}(\mathbf{J}) = 0$ and $R_2 = \det(\mathbf{J}) = 0$ to eliminate square-root branch point singularities and achieve $< 10^{-11}$ residual convergence.

##### `[Table 1]` Specification:
* Complete definitions, algebraic formulas, physical roles, and benchmark coordinates for all 7 dimensionless scaling groups and study continuation points (Point A, Point B, Point C).

---

#### 2.3 Machine Learning Surrogate Architectures, Architectural Rationale, and Statistical Validation Protocols (`[Figure 3]`)

##### Paragraph 1: Problem Formulation, Limit-State Margin, & Sampling Strategy
* *Limit-State Formulation:* Formulating safety evaluation as continuous margin regression:
  $$g(\mathbf{x}) = Eu_{target} - Eu(N_{pch}), \quad \mathbf{x} = (N_{sub}, N_{pch})$$
  where $g(\mathbf{x}) > 0$ denotes safe operation, $g(\mathbf{x}) = 0$ is the safety frontier, and $g(\mathbf{x}) < 0$ is unstable.
* *Latin Hypercube Sampling:* Generating space-filling Latin Hypercube Sampling (LHS) designs across operating parameter domains surrounding Points A, B, and C.
* *Citations:* Limit-state reliability theory and Latin hypercube sampling (Sudret 2012, McKay et al. 1979, Iman & Conover 1980).

##### Paragraph 2: Baseline Architecture Rationale & JAX/Optax Pipeline
* *Why Multi-Layer Perceptrons (MLPs) as the Baseline:*
  - Fully-connected MLPs are the canonical universal approximator (Cybenko 1989, Hornik 1991) and the industry-standard architecture across 90%+ of engineering surrogate models and digital twins in thermal-hydraulics, nuclear engineering, and structural reliability (Echard et al. 2011, Marelli & Sudret 2014, Tripathy & Bilionis 2018).
  - Continuous mapping from low-dimensional operational parameters $\mathbf{x} = (N_{sub}, N_{pch}) \to g(\mathbf{x})$ is natively an MLP regression problem (CNNs and Transformers are structurally unsuited for unstructured continuous scalars).
  - Demonstrating that standard MLPs fail at BT points exposes a critical vulnerability in the most widely deployed class of industrial surrogates.
* *Implementation Pipeline:* 3 hidden layers with 128 units ($3\times128$), SiLU activations; optimized using AdamW (Loshchilov & Hutter 2019) with cosine learning rate schedules in JAX/Optax.

##### Paragraph 3: Systematic 4-Way Remediation Taxonomy & Theoretical Rationales
We construct a principled, 4-way orthogonal taxonomy of candidate fixes addressing every stage of the machine learning pipeline:
1. *Feature Space (Frequency Domain) Fix — Multi-Scale Random Fourier Features (`fourier_mlp.py`):*
   - *Rationale:* Standard MLPs suffer from Spectral Bias (Rahaman et al. 2019, Basri et al. 2020), learning low spatial frequencies first and struggling with high-frequency spatial cusps. Projecting inputs through Gaussian frequency matrices $\sin(\mathbf{B}\mathbf{x}), \cos(\mathbf{B}\mathbf{x})$ enables high-frequency coordinate learning (Tancik et al. NeurIPS 2020, Rahimi & Recht 2007).
2. *Input Space (Coordinate Prior) Fix — Input-Augmented Log-Distance MLP (`log_distance_mlp.py`):*
   - *Rationale:* Injects explicit analytical singularity knowledge by augmenting the input vector with $\ln(\|\mathbf{x} - \mathbf{x}_{BT}\|)$, providing a geometric distance prior to the network.
3. *Data Space (Active Learning) Fix — Misfit-Driven Adaptive Sampling (`adaptive_sampling.py`):*
   - *Rationale:* The classical reliability engineering approach (AK-MCS in RESS: Echard et al. 2011, Bect et al. 2012), iteratively placing dense training samples directly along high-uncertainty boundary zones.
4. *Loss Optimization Space (Landscape Warping) Fix — Inverse-Distance Boundary-Weighted Loss (`boundary_weighted_mlp.py`):*
   - *Rationale:* Restructures the loss landscape directly via:
     $$\mathcal{L}_{BW} = \frac{1}{|g_{true}(\mathbf{x})| + \varepsilon_w} \Big( g_{pred}(\mathbf{x}) - g_{true}(\mathbf{x}) \Big)^2$$
     forcing optimization gradients to concentrate at the safety boundary $g \to 0$, directly overcoming spectral bias (Wang et al. JCP 2021, McClenny & Braga-Neto 2022).

##### Paragraph 4: Evaluation Metrics, The Metric Ratio Fallacy, & Multi-Seed Statistical Validation
* *Exposing the Near/Far Ratio Fallacy:* Proving mathematically why $\text{Ratio} = \text{Error}_{near} / \text{Error}_{far}$ is a deceptive metric that perversely penalizes high-capacity models when far-field accuracy improves faster than near-boundary accuracy.
* *The Certified Metric:* Establishing **Absolute Near-Boundary Root Mean Squared Error** ($\text{RMSE}_{near} = \sqrt{\frac{1}{N_{near}}\sum (g_{pred}-g_{true})^2}$) on points within $|g| \le \varepsilon$ as the mandatory safety certification benchmark.
* *Multi-Seed Statistical Hypothesis Testing:* 10 independent random initialization seeds evaluated via paired, two-tailed non-parametric **Wilcoxon signed-rank hypothesis tests** ($p < 0.005$) to establish rigorous statistical significance over baseline models (Wilcoxon 1945, Demšar JMLR 2006).

---

### 3. Results and Discussion

#### 3.1 Ground-Truth Bifurcation Topologies and Stable Operating Window Scaling (`[Figure 3]`)

##### Paragraph 1: Physical Reality of the Stable Operating Window Across Parameter Topologies
* *Physical Meaning of the Stable Operating Envelope:* In an operating boiler tube or nuclear fuel assembly, the shaded region between the lower Fold curve and upper Hopf curve represents the permissible operating envelope where boiling flow remains stable. Operating below the lower Fold boundary triggers static Ledinegg flow collapse; operating above the upper Hopf boundary triggers dynamic Density-Wave Oscillations.
* *Point A (Transversal Intersection):* Operating at realistic facility conditions ($\Lambda = 5.90, Fr = 0.035, k_{in} = 6.55, k_{out} = 2.03$). The Fold and Hopf boundaries cross at a distinct physical angle ($\Delta\text{slope} \approx 0.92$). The intersection is topologically non-degenerate (serving as the negative control).
* *Points B & C (Bogdanov–Takens Degeneracies):* Operating at $\Lambda = 0.001$. As inlet water gets colder (moving toward $N_{sub,BT}$), the two boundaries converge and touch tangentially ($\Delta\text{slope} = 0$). Beyond the BT vertex ($N_{sub} > N_{sub,BT}$), **no stable operating state exists at any heating power**—the system is physically guaranteed to either collapse or oscillate.

##### Paragraph 2: Physical Coalescence of Instability Modes & Machine-Precision Continuation
* *Double-Zero Eigenvalue Coalescence ($\mu_1 = \mu_2 = 0$):* Physically represents the coalescence of the non-oscillatory excursive mode ($\mu_1 = 0$) and the oscillatory acoustic transit mode ($\mathrm{Re}(\mu) = 0, \mathrm{Im}(\mu) \to 0$) into a single degenerate physical state.
* *Machine-Precision Verification:* Point B located at $(N_{sub} = 14.142794816, N_{pch} = 20.597778032)$ and Point C at $(N_{sub} = 7.630101792, N_{pch} = 10.913202566)$ with polynomial residual $< 10^{-11}$. Dynamic Jacobian tracking confirms continuous spectral migration from stable focus to unstable limit cycles.

##### Paragraph 3: Quantitative Window Narrowing & The "Knife-Edge" Stability Corridor ($\delta^{1.00}$)
* *Physical Significance of Linear Narrowing:* Direct multi-decade evaluation over $\delta \in [10^{-4}, 1.0]$ confirms exact linear power-law scaling:
  $$\text{Window Width } \Delta N_{pch} = N_{pch,upper} - N_{pch,lower} \propto \delta^{\mathbf{1.00}} \quad (R^2 = 0.99999985)$$
* *The Engineering Danger:* Proves that near the BT point, the permissible operating heating margin shrinks into an ultra-narrow "knife-edge" corridor where a tiny $0.1\%$ fluctuation instantly triggers violent instability.

---

#### 3.2 The Metric Decoupling Phenomenon and Diagnostic Root-Cause Analysis (`[Figure 4]`, `[Figure 5]`)

##### Paragraph 1: The False Illusion of Global Accuracy ($R^2 > 0.999$)
* *The Numerical Trap:* Training a standard $3\times128$ MLP yields aggregate metrics that appear flawless ($R^2 > 0.999$, global $\text{RMSE} < 0.01$). Parity plots show tight diagonal clustering.
* *The Regulatory Hazard:* Under standard industrial verification protocols, inspecting only global aggregate statistics would mistakenly certify this surrogate as production-ready.

##### Paragraph 2: The Decoupling Discovery Across Topologies (Point A vs. Points B & C)
* *Point A (Transversal Crossing - Negative Control):* Uniformly low error across far-field ($0.010$) and near-boundary ($0.012$) domains. The network reconstructs the non-degenerate crossing with near-zero distortion.
* *Points B & C (Bogdanov–Takens Degeneracies):* Despite identical global $R^2 > 0.999$, near-boundary error **explodes by $3.5\times\text{--}10\times$** ($0.083 \pm 0.011$ at Point B, $0.112 \pm 0.014$ at Point C).
* *Physical Safety Risk:* The surrogate distorts the narrow stability wedge, falsely predicting that unstable operating states are safe, which would lead to unpredicted flow excursions or burnout in an operating plant.

##### Paragraph 3: Diagnostic Root-Cause Investigation (Refuting Gradient Vanishing & Proving Spectral Bias)
* *Diagnostic 1 (Gradient Landscapes):* Direct measurement across thousands of points confirms loss gradients do not vanish ($\|\nabla g\| = 1.34 \pm 0.12$ far vs. $1.31 \pm 0.15$ near). The failure is not an optimization defect.
* *Diagnostic 2 (Spectral Bias):* Standard MLPs act as low-pass spatial filters (Rahaman et al. 2019). Because the safe corridor narrows linearly ($\delta^{1.00}$), the area inside the cusp shrinks to sub-mesh dimensions. Minimizing global MSE forces the network to "blur" over the sharp tangent cusp to optimize the vast outer regions.

---

#### 3.3 Comparative Performance of Remediation Strategies and Statistical Validation (`[Figure 6]`, `[Figure 7]`, `[Table 2]`)

##### Paragraph 1: Quantitative Performance Across the 4 Remediation Frameworks
* *Baseline MLP ($3\times128$):* Point B $\text{RMSE}_{near} = 0.083 \pm 0.011$, Point C $= 0.112 \pm 0.014$. Fails because standard unweighted MSE smooths over the needle-sharp cusp.
* *Random Fourier Features:* Point B $= 0.076 \pm 0.009$ ($-8\%$), Point C $= 0.104 \pm 0.012$ ($-7\%$). High-frequency projections help slightly globally, but lack localized spatial focus along the boundary.
* *Input-Augmented Log-Distance MLP:* Point B $= 0.069 \pm 0.008$ ($-17\%$), Point C $= 0.091 \pm 0.010$ ($-19\%$). Distance prior $\ln(\|\mathbf{x} - \mathbf{x}_{BT}\|)$ guides the network near singular coordinates, but static coordinate priors cannot dynamically adapt to the complex curvature of both branches simultaneously.
* *Misfit-Driven Adaptive Sampling:* Adding dense samples along the boundary yields only $\sim 12\%$ reduction because spectral bias restricts spatial resolution regardless of local point density.
* *Boundary-Weighted Loss Function ($1/(|g|+\varepsilon_w)$):* Point B $= 0.036 \pm 0.004$ (**$-57\%$ error reduction**), Point C $= 0.050 \pm 0.006$ (**$-55\%$ error reduction**). Decisive winner! Dynamically forces gradient descent to prioritize zero-level set accuracy, completely recovering the sharp tangent cusp.

##### Paragraph 2: Multi-Seed Statistical Significance Testing (`[Table 2]`)
* *Multi-Seed Testing Protocol:* 10 independent random initialization seeds per architecture.
* *Paired Non-Parametric Wilcoxon Signed-Rank Hypothesis Tests:*
  - Boundary-Weighted Loss vs. Baseline MLP: $W = 0, p = 0.00195 < 0.005$ (10 out of 10 seeds show statistically significant, uniform improvement).
  - Boundary-Weighted Loss vs. Fourier Features: $W = 0, p = 0.00195 < 0.005$.
  - Boundary-Weighted Loss vs. Log-Distance MLP: $W = 1, p = 0.00391 < 0.005$.
* *Empirical Cumulative Distribution Functions (ECDFs):* The entire error distribution shifts leftward toward zero, proving Boundary-Weighted Loss is robust and seed-invariant.

##### Paragraph 3: Pareto Frontier & Physical Engineering Trade-Off Analysis
* *The Physics-Informed Engineering Trade-off:* In safety-critical thermal systems, far-field error has zero operational safety consequence. Boundary-Weighted Loss accepts a minor, harmless increase in far-field RMSE ($0.008 \to 0.014$) in exchange for cutting critical boundary localization error by more than half ($0.083 \to 0.036$). Pareto curve demonstrates Boundary-Weighted Loss strictly dominates the safety-critical Pareto frontier.

##### `[Table 2]` Specification:
* Complete statistical table reporting: Architecture, Near-Boundary RMSE (mean $\pm$ std), Far-Field RMSE, Error Reduction %, Wilcoxon $W$-statistic, and $p$-values across Points B and C.

---

#### 3.4 Reliability Certification Framework and Industrial Implications (`[Figure 8]`)

##### Paragraph 1: The Four Golden Rules for AI Surrogate Certification in Safety Infrastructure
Synthesizing discoveries into 4 actionable rules for safety regulators (NRC, IAEA, FAA, ASME):
1. *Rule 1 (Prohibition of Global $R^2$ Alone):* Global $R^2 > 0.99$ must never be used as sole proof of safety; high aggregate fidelity consistently masks localized boundary collapse.
2. *Rule 2 (Pre-Deployment Bifurcation Continuation):* Industrial teams must run numerical continuation to identify all Codimension-2 BT points, tangencies, and double-zero eigenvalues in the operational parameter space before deploying digital twins.
3. *Rule 3 (Mandatory Boundary-Weighted Training):* Training objective functions must enforce inverse-distance boundary weighting ($1/(|g|+\varepsilon_w)$) whenever the operational envelope contains tangent cusps.
4. *Rule 4 (Absolute Near-Boundary Error Certification):* Models must be certified on Absolute Near-Boundary RMSE ($\text{RMSE}_{near}$) across multi-seed non-parametric hypothesis tests ($p < 0.01$).

##### Paragraph 2: Industrial Licensing and Digital Twin Verification Workflow (`[Figure 8]`)
A step-by-step decision flowchart for engineering teams building digital twins for nuclear reactors, industrial boilers, and GPU microchannel coolers: Model specification $\to$ Continuation singularity screening $\to$ Boundary-Weighted training $\to$ Multi-seed absolute error thresholding $\to$ Certification for control room deployment.

##### Paragraph 3: Model Scope, Limitations, and Multi-Physics Generalizability
* *Single-Channel 1D Approximation vs. Full Natural-Circulation Loops:* In our 1D single-channel testbed, low friction ($\Lambda = 0.001$) was needed to achieve double-zero eigenvalues due to the omission of external loop components. In full natural-circulation loops with risers and downcomers (Pandey & Singh 2017), genuine Bogdanov–Takens points exist at normal, realistic friction ($\Lambda \approx 3\text{--}6$).
* *Universality of the Failure Mode:* Because surrogate failure is driven by the differential geometry of the tangent cusp (linear window narrowing $\delta^{1.00}$), the failure mode and the Boundary-Weighted Loss remedy apply universally across all two-phase systems, aeroelastic flutter boundaries in aerospace, and runaway reaction boundaries in chemical engineering.

---

### 4. Conclusions
* **Paragraph 1: Summary of Core Scientific Discoveries:** Recapitulation of the metric decoupling phenomenon, the failure of global $R^2$ at codimension-2 Bogdanov–Takens points, and the spectral bias representability barrier on linear tangent cusps ($\delta^{1.00}$).
* **Paragraph 2: Engineering Remediation & Statistical Superiority:** Summary of the Boundary-Weighted Loss function, which achieves a statistically verified $\sim 57\%$ error reduction at Point B and $\sim 55\%$ at Point C ($p < 0.005$) while strictly dominating the safety-critical Pareto frontier.
* **Paragraph 3: Future Research Directions:** Recommendations for extending boundary-weighted training to subcooled boiling models, coupled 3D CFD digital twins, and Physics-Informed Neural Operators (PINOs) in nuclear and energy infrastructure.

---

### Appendix / Supplementary Material

#### Appendix A: Complete Algebraic Derivation of the Clausse–Lahey Momentum ODE System
* **A.1 Single-Phase Enthalpy Integration & Kinematic Boundary Equation:**
  * Step-by-step Leibniz integration of $\frac{\partial (\rho h)}{\partial t} + \frac{\partial (\rho u h)}{\partial z} = q'' \frac{P_h}{A_c}$ from $z = 0$ to $z = \lambda(t)$.
  * Derivation of the nodal recurrence relation for internal nodes $l_n(t)$ ($n = 1, \dots, N_1$).
* **A.2 Two-Phase Mixture Density & Mass Inventory Integration:**
  * Step-by-step integration of continuity equation $\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial z} = 0$ across $\lambda(t) \le z \le 1$.
  * Full algebraic derivation of $(1-\lambda)\frac{\ln(r)}{r-1} = m - \lambda$ and time-derivative $\dot{\rho}_e$.
* **A.3 Momentum Equation & Analytical Bracket Expansion:**
  * Spatial integration of momentum PDE from $z = 0$ to $1$.
  * Analytical expansion of Momentum Accumulation ($u_i \dot{m} + \text{Bracket}$), Accelerational drop ($\rho_e u_e^2 - u_i^2$), Gravity ($m/Fr$), Wall Friction ($\text{Friction}(\Lambda)$), and Orifice Losses ($k_{in} u_i^2 + k_{out} \rho_e u_e^2$).
  * Combining into explicit liquid acceleration ODE: $\frac{du_i}{dt} = -\frac{1}{m}[u_i \dot{m} + \text{Bracket} + \text{Rest} + \text{Friction}]$.
* **A.4 Derivation of the Closed-Form Steady-State Euler Characteristic (Eq. 26 in `ledinegg_curve.py`):**
  * Steady-state integration setting all time derivatives to zero, resulting in the 4-term analytical Euler formula $Eu(N_{pch}, N_{sub})$.

#### Appendix B: Numerical Continuation Algorithms, Solver Verification, and Nodal Convergence
* **B.1 Pseudo-Code for Automatic Differentiation Gradient Bisection (Fold Curve Solver):** Exact bisection algorithm on $\frac{\partial Eu}{\partial N_{pch}} = 0$ using `jax.grad`.
* **B.2 Pseudo-Code for Dynamic Jacobian Eigenvalue Bisection (Hopf Curve Solver):** Exact algorithm computing $\mathbf{J} = \left.\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right|_{\mathbf{x}^*}$ via `jax.jacfwd`, sorting complex eigenvalues, and bisecting on $\mathrm{Re}(\mu) = 0$.
* **B.3 The 2D Newton Sum-and-Product Algorithm for Bogdanov–Takens Points:** 2D Newton formulation on polynomial invariants $R_1 = \mathrm{Tr}(\mathbf{J}) = 0$ and $R_2 = \det(\mathbf{J}) = 0$ with analytical Jacobian updates.
* **B.4 Nodal Independence and Verification Across Node Counts ($N_1 = 2, 4, 8, 16$):** Convergence tables showing steady states, continuation coordinates, and eigenvalues as $N_1$ increases from $2 \to 16$, and verifying $< 10^{-15}$ machine-precision consistency with Eq. (26).

#### Appendix C: Hyperparameter Specifications, Training Configurations, and Extended Statistical Tables
* **C.1 Complete Neural Network Hyperparameter Specifications:** Layer dimensions, activation functions, learning rates, cosine decay schedules, AdamW weight decay, batch sizes, epochs, and initialization seeds.
* **C.2 Multi-Scale Random Fourier Features Matrix Generation:** Gaussian projection matrix distribution $\mathbf{B} \sim \mathcal{N}(0, \sigma^2)$ and frequency bandwidth tuning.
* **C.3 Complete 10-Seed Statistical Data Tables:** Full individual seed-level performance tables for Baseline MLP, Random Fourier Features, Log-Distance MLP, and Boundary-Weighted Loss across Points B and C, including raw test RMSE, $R^2$, and exact Wilcoxon signed-rank test calculation logs ($W$-statistic, rank sums, exact two-tailed $p$-values).
