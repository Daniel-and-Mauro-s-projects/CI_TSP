import numpy as np
import random
import matplotlib.pyplot as plt
import logging

from parent_selection import parent_selection as ps
from mutation import mutation as mu
from crossover import crossover as cr

class TSP_Genetic:
    """Perform a Genetic Algorithm simulation for the TSP problem"""

    def __init__(
        self,
        generations: int=100,
        print_rate: int=10,
        m_rate: float=0.05,
        c_rate: float=0.8,
        select_parents: str="tournament_selection",
        tournament_size: int=3,
        crossover: str="OX1",
        mutation: str="insertion",
        elitism: int=0
    ):

        """Initialize the Genetic Algorithm simulation for the TSP problem.
            Args:
                generations: The number of generations to simulate in the genetic algorithm (100 by default).
                print_rate: How often to print the progress of the algorithm (Default: every 10 generations).
                m_rate: The mutation rate, representing the probability of mutation in offspring (Default: 0.05).
                c_rate: The crossover rate, representing the probability of crossover between parent chromosomes (Default: 0.8).
                select_parents: The method to select parents for crossover. Options are 'tournament_selection' and 'roulette_selection' (Default: 'tournament').
                tournament_size: The size of the tournament for tournament selection (Default: 3).
                crossover: The method to perform crossover between parent chromosomes.
                mutation: The method to perform mutation in offspring chromosomes.
                elitism: The number of best chromosomes to keep in the next generation (Default: 0).
        """
        self.generations = generations
        self.print_rate = print_rate

        self.m_rate = m_rate
        self.c_rate = c_rate

        self.select_parents = select_parents
        self.tournament_size = tournament_size
        self.crossover = crossover
        self.mutation = mutation
        self.elitism = int(elitism)

    def _fitness(self,x):
        """Calculate the fitness of a chromosome x for the TSP problem.
            Args:
                x: The chromosome to evaluate.
            Returns:
                The fitness of the chromosome.
        """
        fitness = 0
        for i in range(len(x)-1):
            fitness += self.distances[x[i],x[i+1]]
        fitness += self.distances[x[-1],x[0]]
        return fitness
    
    def fitness_population(self,population):
        """Calculate the fitness of a population of chromosomes for the TSP problem.
            Args:
                population: The population to evaluate.
            Returns:
                The fitness of each chromosome in the population.
        """
        return np.array([self._fitness(x) for x in population])
    
    def plot_route(self,z):
        """Plot the route of a chromosome z for the TSP problem.
            Args:
                z: The chromosome to plot.
        """
        # Plot the cities
        plt.plot(self.cities[:,0],self.cities[:,1],'o-')
        for (x, y), label in zip(self.cities, [str(i) for i in range(1, self.n_cities+1)]):
            plt.text(x, y + 0.1, label, ha='center', va='bottom')

        # Plot the route
        for i in range(len(z)-1):
            plt.plot([self.cities[z[i],0],self.cities[z[i+1],0]],[self.cities[z[i],1],self.cities[z[i+1],1]],'r-')

        plt.show()


    def run(self,population,cities,distances):
        """Run the Genetic Algorithm simulation for the TSP problem.
            Args:
                population: The initial population of chromosomes (list of lists).
                cities: The list of positions of each city. A row represents a city and the first element its x coordinate and the second its y coordinate.
                distances: A matrix representing the distances between each pair of cities.
        """
        self.population = population
        self.population_size = len(self.population)
        self.cities = np.array(cities)
        self.n_cities = self.cities.shape[0]
        self.distances = np.array(distances)

        # We need population_size-elitism to be even
        if (self.population_size-self.elitism) % 2 != 0:
            logging.warning("Population size - elitism is not even. Adding one to population size.")
            self.population_size += 1

        for generation in range(self.generations):
            # Get fitness of a population
            fitness = self.fitness_population(self.population)
            offspring = []
            #Elitism
            if self.elitism > 0:
                for indx in np.argsort(fitness)[:self.elitism]:
                    offspring.append(self.population[indx])
            
            # Generate offspring:
            for i in range(int((self.population_size-self.elitism)/2)):
                # Select parents
                if self.select_parents == "tournament_selection":
                    parent1 = ps.tournament_selection(self.population,fitness,self.tournament_size)
                    parent2 = ps.tournament_selection(self.population,fitness,self.tournament_size)
                else:
                    parent1 = getattr(ps, self.select_parents)(self.population,fitness)
                    parent2 = getattr(ps, self.select_parents)(self.population,fitness)
                
                #Crossover
                if random.random() < self.c_rate:
                    # Generate offspring
                    child1, child2 = getattr(cr(parent1,parent2), self.crossover)()
                else:
                    child1, child2 = parent1, parent2

                # Mutate offspring
                # TODO: Write the mutation function
                if random.random() < self.m_rate:
                    child1 = getattr(mu,self.mutation)(child1)
                if random.random() < self.m_rate:
                    child2 = getattr(mu,self.mutation)(child2)

                offspring.append(child1,child2)

            population = np.array(offspring)
            # Print progress
            if generation-1 % self.print_rate == 0:
                logging.INFO(f"Generation {generation} - Best cromosome: {population[np.argmin(fitness)]}, Best fitness: {np.min(fitness)}")
                self.plot_route(population[np.argmin(fitness)])
        
        return self.population[0]