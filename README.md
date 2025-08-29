# Problem Set for 2025 Fall ES/AM 158

This is the P-SET repository for the 2025 Fall ES/AM 158 class, **Introduction to Optimal Control and Reinforcement Learning**, at Harvard University.

- Course site (Canvas): <https://canvas.harvard.edu/courses/153422>  

- Lecture notes: <https://hankyang.seas.harvard.edu/OptimalControlReinforcementLearning/>

- Syllabus: <https://docs.google.com/document/d/1dIRYQZZJDx8K2q1TrodDDLg-bKJWWmj7o7yzOGlIs7o/edit?usp=sharing/>

All problem sets are provided as Jupyter notebooks (`.ipynb`).  
- **Pen-and-paper items:** fill your answers in the designated blank cells.  
- **Coding items:** complete the `TODO` blocks and **run all cells** so outputs are visible.

**Submission.** Submit via **Gradescope**. Upload a **single PDF** exported from your `.ipynb` with all outputs shown.

---

## Prerequisites

You should be comfortable with the topics below. If not, you can **self-study the relevant background with P-SET 0 and the refresher links below**. Always ask ChatGPT if you have problem debugging.

**Linear Algebra**
- Vectors/matrices, norms & inner products, eigen/SVD, least squares. 

**Calculus**
- Gradients/Jacobians/Hessians; basic integration.

**Probability / Statistics**
- Probability basics, Bayes’ rule, expectation/variance/covariance, Gaussian distribution.

**Optimization**
- Convex sets/functions, first-order optimality, gradient descent & backtracking.  
- Lecture note: <https://hankyang.seas.harvard.edu/OptimalControlReinforcementLearning/appconvex.html>
- Book: [*Convex Optimization* — Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

**Python / Jupyter / NumPy**
- Notebooks, vectorization/broadcasting, scientific computing, plotting.  
- Python crash course: <https://fgnt.github.io/python_crashkurs/#/>
- Numpy Quick start: <https://numpy.org/doc/stable/user/quickstart.html>

**LaTeX**
- Math symbols like $x^y$, $\int$, $\phi$; equation environments.  
- Quick start: <https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes#Adding_math_to_LaTeX>

---

## Setting environment

### Colab environment (Recommended)

Open the repository in Colab (lists notebooks in the repo):
<https://colab.research.google.com/github/ComputationalRobotics/2025-ES-AM-158-PSET>

**Minimal setup cell (put at the top of your notebook):**
```python
# Install runtime dependencies (bound to the current kernel)
%pip install numpy matplotlib tqdm gymnasium cvxpy
```

### Local python environment 
Python version: 3.10
```
conda create -n 2025ocrl python=3.10
conda activate 2025ocrl
pip install -r requirements.txt
```