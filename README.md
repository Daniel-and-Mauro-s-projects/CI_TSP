# CI_TSP
Genetic Algorithms+TSP

# To do list
- [ ] Add a different method apart from elistism
- [ ] Add hyperparameters:
  - [ ] nº of iterations
  - [ ] nº of prints (how often we represent the evolution on the fitness of the generations)
- [ ] Add matrix of distances implementation
- [ ] Add crossover methods (maybe create our own)
- [ ] Improve computing time

# Optional to do list:
- [ ] We could do a class for every element in the population with the orther and the distances
- [ ] To test if we chose the right parameters we can repeat a certain time the experiments and perform an ANOVA on the different models (this is done in the article [3])
- [ ] We could add stagnation convergence
- [ ] Add replacement strategy

## Info for the report
* We use the **Pittsburgh view**, each individual is a full solution.
* 

## Selection Methods:
* Elitism
* Roulette Wheel (based on fitness)
* Roulette Wheel (based on rank)
* Niching (Doesn't seem ideal for this problem)
  
## Replacement strategies:
* Generational (offspring replace all of the parents)
* Offspring replaces the wors
* Offspring replaces at random
* Offspring replaces the oldest

## Crossover Posibilities
After looking into other modern papers, we have more than enough with the crossover posibilies proposed at [1].

## 


# Bibliography:
1. *Genetic Algorithms for the Travelling Salesman Problem: A Review of Representations and Operators*, P. Larrañaga et al, (1999)
2. Github repository with a basic implementation: "https://github.com/hassanzadehmahdi/Traveling-Salesman-Problem-using-Genetic-Algorithm/"
3. *The Effect of Genetic Algorithm Parameters Tuning for Route Optimization in Travelling Salesman Problem through General Full Factorial Design Analysis* (2022)
  

https://github.com/hassanzadehmahdi/Traveling-Salesman-Problem-using-Genetic-Algorithm/tree/main