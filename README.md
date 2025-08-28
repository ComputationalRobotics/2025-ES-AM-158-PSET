# Problem set for 2025 fall ES/AM 158

This is the P-SET repository for 2025 fall ES/AM 158 class [Introduction to Optimal Control and Reinforcement](https://canvas.harvard.edu/courses/153422). Lecture note can be found [here](https://hankyang.seas.harvard.edu/OptimalControlReinforcementLearning/). 

All the problem are written in Jupyter notebook `.ipynb` file. For pen and paper problem, you need to fill in the answer in blank cells. For coding problem complete the `TODO` paragraph and left the output in notebook.

**Submission**: Through gradescope, link TBD. You should submit your `.ipynb` file as a pdf and include all the output.

---
### Prerequisites

You should be comfortable with the topics below. If not, you can self-check in P-SET 0 and .

**Linear Algebra**
- Vectors/matrices, norms & inner products, eigen/SVD, least squares. 

**Calculus**
- Gradients/Jacobians/Hessians; basic integration  

**Probability / Statistics**
- Probability basics, Bayes’ rule, expectation/variance/covariance, Gaussian distribution; 
  
**Optimization**
- Convex sets/functions, first-order optimality, gradient descent & backtracking; 
- Book: [*Convex Optimization, Stephen Boyd*](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

**Python / Jupyter / NumPy**
- Notebooks, vectorization/broadcasting, scientific computing, plotting 
- [Numpy quick start](https://numpy.org/doc/stable/user/quickstart.html) 

**$\LaTeX$**
- Math symbols like $x^y$,$\int$,$\phi$. Equations.
- [Latex quick start](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes#Adding_math_to_LaTeX)

---
### Setting environment

#### Colab environment (Recommended)
Open the repository with [Colab](https://colab.research.google.com/github/ComputationalRobotics/2025-ES-AM-158-PSET
)
```
!pip install numpy matplotlib tqdm
```

#### Local python environment
Python version: 3.10
```
conda create -n 2025ocrl python=3.10
conda activate 2025ocrl
pip install -r requirements.txt
```
