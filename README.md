# CI_TSP
Genetic Algorithms+TSP

# Approach Doubts:
- Should we save all the fitness (maybe for plots)
- Should we create a number of offspring and then select between them and the parents
  - If we do this how do we do it. We need two parents per offspring (maybe select even offspring values). If there's no crossover just keep the values? Doesn't this account too many times for the parents
- Should we just create offspring and select from there

# To do list
- [x] Elitism
- [ ] Plot function
- [ ] Add hyperparameters:
  - [x] nº of iterations
  - [x] nº of prints (how often we represent the evolution on the fitness of the generations)
- [x] Add matrix of distances implementation
- [ ] Improve computing time
- [ ] Code parent selection methods
  - [x] tournament
  - [x] roulette
- [x] Code crossover methods
  - [x] OX1
  - [x] POS
- [x] Code mutation methods 
  - [x] Insertion
  - [x] Exchange
  - [x] IVM
- [ ] Add stagnation convergence (before ending the generations)
- [ ] AT THE END
  - [ ] Check function descriptors says all the hyperparameter options

# Optional to do list:
- [ ] To test if we chose the right parameters we can repeat a certain time the experiments and perform an ANOVA on the different models (this is done in the article [3])
- [ ] We could add stagnation convergence
- [ ] Add replacement strategy

## Info for the report
* We use the **Pittsburgh view**, each individual is a full solution.
* We assume simmetrical distances (since the data we obtained was that way)
* we choose $(\mu,\lambda)$ with $\mu=\lambda$

## Selection Methods:
* Elitism
* Roulette Wheel (based on fitness)
* Roulette Wheel (based on rank)
* Niching (Doesn't seem ideal for this problem)
  
## Replacement strategies:
* Generational (offspring replace all the parents)
* Offspring replaces the worse
* Offspring replaces at random
* Offspring replaces the oldest

## Crossover Possibilities
After looking into other modern papers, we have more than enough with the crossover possibilities proposed at [1].

## 


# Bibliography:
1. *Genetic Algorithms for the Travelling Salesman Problem: A Review of Representations and Operators*, P. Larrañaga et al, (1999)
2. Github repository with a basic implementation: "https://github.com/hassanzadehmahdi/Traveling-Salesman-Problem-using-Genetic-Algorithm/"
3. *The Effect of Genetic Algorithm Parameters Tuning for Route Optimization in Travelling Salesman Problem through General Full Factorial Design Analysis* (2022)
4. We obtained the data from http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/
5. 
  

https://github.com/hassanzadehmahdi/Traveling-Salesman-Problem-using-Genetic-Algorithm/tree/main