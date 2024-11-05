# CI_TSP
Genetic Algorithms+TSP

# To do list
- [ ] Add a different method apart from elitism
- [ ] Add hyperparameters:
  - [x] nº of iterations
  - [x] nº of prints (how often we represent the evolution on the fitness of the generations)
- [x] Add matrix of distances implementation
- [ ] Add crossover methods (maybe create our own)
- [ ] Improve computing time
- [ ] Code parent selsction methods
  - [ ] tournament
  - [ ] roulette
- [ ] Code crossover methods
- [ ] Code mutation methods (?)
- [ ] Code select survivors methods
- [ ] Add stagnation convergence (before ending the generations)

# Optional to do list:
- [ ] To test if we chose the right parameters we can repeat a certain time the experiments and perform an ANOVA on the different models (this is done in the article [3])
- [ ] We could add stagnation convergence
- [ ] Add replacement strategy

## Info for the report
* We use the **Pittsburgh view**, each individual is a full solution.
* We assume simmetrical distances (since the data we obtained was that way)
* experimentally lambda = 7 x mu

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