# EHVI-HVI
Expected hypervolume improvement (EHVI) as a hypervolume improvement

EHVI is an important concept in multi-objective Bayesian optimization. This is a Python implementation of our formulation for exactly computing EHVI in [1]. The experimental results have shown its significant advantages over previously fastest implementation in [2] for $d\ge3$. 

Our implementation also works well for exactly computing $q\mathrm{EHVI}$ (multi-point version of EHVI) within the same time complexity and the gradient of EHVI by using the method in [3].

# Technical Appendix
The appendix provides proofs and experimental results of our work [1].

# Dependency
BoTorch: https://github.com/pytorch/botorch

# TODO
C++, Matlab implementations and Python implementation independent from BoTorch

# References
[1] Deng, J.; Sun, J.; Zhang, Q.; Li, H. 2025. Expected Hypervolume Improvement Is a Particular Hypervolume Improvement. AAAI 2025: 16217-16225. DOI: 10.1609/aaai.v39i15.33781.

[2] Balandat, M.; Karrer, B.; Jiang, D. R.; Daulton, S.; Letham, B.;Wilson, A. G.; and Bakshy, E. 2020. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 2020.

[3] Deutz, A.; Emmerich, Michael T. M.; Wang, H. The Hypervolume Indicator Hessian Matrix: Analytical Expression, Computational Time Complexity, and Sparsity. EMO 2023: 405-418.

# Contact
Jingda Deng

School of Mathematics and Statistics

Xi’an Jiaotong University

Email: ~jddeng@xjtu.edu.cn~ (invalid)

jddeng@xaut.edu.cn
