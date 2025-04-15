# EHVI-HVI
Expected hypervolume improvement (EHVI) as a hypervolume improvement

EHVI is an important concept in multi-objective Bayesian optimization. This is a Python implementation of our work in [1]. This algorithm has a complexity of $\tilde{O}(n^{\frac{d}{3}})$ ($n$ is the number of points and $d$ is the number of objectives) for \textbf{exactly} computing EHVI for all $d\ge4$, which improves previously fastest implementation in [2] by $\tilde{O}(n^{\frac{d}{6}})$. The experimental results show significant advantages over [2] for $d\ge5$. 

Our implementation is also suitable for computing the gradient of EHVI, by applying the method in [3].

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
