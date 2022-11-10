# Cuda Sudoku Solver
A sudoku solver created using Nvidia's CUDA parallel computing platform. 
Solves around 2.7 million sudoku boards per second which is in the top 5% of solutions in my operating systems class at Grinnell College.
The speed of the solver mainly comes from using multiple cores in the GPU to solve many boards at once and from using `__shared__` memory within the thread blocks to speed up memory accesses.
