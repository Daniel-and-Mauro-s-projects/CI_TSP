import numpy as np
import random
import matplotlib.pyplot as plt
import logging

from src.parent_selection import parent_selection as ps
from src.mutation import mutation as mu
from src.crossover import crossover as cr

import json

logging.basicConfig(level=logging.INFO)

class TSP_Genetic:
    """Perform a Genetic Algorithm simulation for the TSP problem"""

    def __init__(
        self,
        generations: int=100,
        print_results: bool=False,
        print_rate: int=10,
        m_rate: float=0.05,
        c_rate: float=0.8,
        select_parents: str="tournament_selection",
        tournament_size: int=3,
        crossover: str="OX1",
        crossover_call:str ="(4, cities.shape[0] - 4)",
        mutation: str="insertion",
        elitism: int=0,
        results_path: str = "results/path_distance_statistics.json",
        save_results: bool = 0
    ):
        """Initialize the Genetic Algorithm simulation for the TSP problem.
            Args:
                generations: The number of generations to simulate in the genetic algorithm (100 by default).
                print_results: Whether to print the results of the algorithm (default: False).
                print_rate: How often to print the progress of the algorithm (Default: every 10 generations).
                m_rate: The mutation rate, representing the probability of mutation in offspring (Default: 0.05).
                c_rate: The crossover rate, representing the probability of crossover between parent chromosomes (Default: 0.8).
                select_parents: The method to select parents for crossover. Options are 'tournament_selection' and 'roulette_selection' (Default: 'tournament').
                tournament_size: The size of the tournament for tournament selection (Default: 3).
                crossover: The method to perform crossover between parent chromosomes.
                crossover_call: The arguments to pass to the crossover method.
                mutation: The method to perform mutation in offspring chromosomes.
                elitism: The number of best chromosomes to keep in the next generation (Default: 0).
                results_path: The path to save the path_distance statistics. (default: "path_distance_statistics.json")
                save_results: Whether to save the path_distance statistics to a file. (default: False)
        """
        self.generations = generations
        self.print_results = print_results
        self.print_rate = print_rate
        self.results_path = results_path
        self.save_results = save_results

        self.m_rate = m_rate
        self.c_rate = c_rate

        self.select_parents = select_parents
        self.tournament_size = tournament_size
        self.crossover = crossover
        self.crossover_call = crossover_call
        self.mutation = mutation
        self.elitism = int(elitism)

    def _path_distance(self,x):
        """Calculate the path_distance of a chromosome x for the TSP problem.
            Args:
                x: The chromosome to evaluate.
            Returns:
                The path_distance of the chromosome.
        """
        path_distance = 0
        for i in range(len(x)-1):
            path_distance += self.distances[x[i],x[i+1]]
        path_distance += self.distances[x[-1],x[0]]
        return path_distance
    
    def path_distance_population(self,population):
        """Calculate the path_distance of a population of chromosomes for the TSP problem.
            Args:
                population: The population to evaluate.
            Returns:
                The path_distance of each chromosome in the population.
        """
        return np.array([self._path_distance(x) for x in population])
    
    def plot_route(self,z):
        """Plot the route of a chromosome z for the TSP problem.
            Args:
                z: The chromosome to plot.
        """
        # Plot the cities
        plt.plot(self.cities[:,0],self.cities[:,1],'o')
        for (x, y), label in zip(self.cities, [str(i) for i in range(1, self.n_cities)]):
            plt.text(x, y + 0.1, label, ha='center', va='bottom')

        # Plot the route
        for i in range(len(z)-1):
            plt.plot([self.cities[z[i],0],self.cities[z[i+1],0]],[self.cities[z[i],1],self.cities[z[i+1],1]],'r-')
        plt.plot([self.cities[z[len(z)-1],0],self.cities[z[0],0]],[self.cities[z[len(z)-1],1],self.cities[z[0],1]],'r-')
        plt.show()


    def run(self,population,cities,distances):
        """Run the Genetic Algorithm simulation for the TSP problem.
            Args:
                population: The initial population of chromosomes (list of lists).
                cities: The list of positions of each city. A row represents a city and the first element its x coordinate and the second its y coordinate.
                distances: A matrix representing the distances between each pair of cities.

            Returns:
                The best chromosome found and its path_distance.
        """
        self.population = population
        self.population_size = len(self.population)
        self.cities = np.array(cities)
        self.n_cities = self.cities.shape[0]
        self.distances = np.array(distances)

        # We create a variable to save the metrics
        data = []

        # We need population_size-elitism to be even
        if (self.population_size-self.elitism) % 2 != 0:
            logging.warning("Population size - elitism is not even. Adding one to population size.")
            self.population_size += 1

        for generation in range(self.generations):
            # Get path_distance of a population
            path_distance = self.path_distance_population(self.population)
            offspring = []
            #Elitism
            if self.elitism > 0:
                for indx in np.argsort(path_distance)[:self.elitism]:
                    offspring.append(self.population[indx])
            
            # Generate offspring:
            for i in range(int((self.population_size-self.elitism)/2)):
                # Select parents
                if self.select_parents == "tournament_selection":
                    parent1 = ps.tournament_selection(self.population,path_distance,self.tournament_size)
                    parent2 = ps.tournament_selection(self.population,path_distance,self.tournament_size)
                else:
                    parent1 = getattr(ps, self.select_parents)(self.population,path_distance)
                    parent2 = getattr(ps, self.select_parents)(self.population,path_distance)
                
                #Crossover
                if random.random() < self.c_rate:
                    # Generate offspring
                    cross_method= getattr(cr(parent1,parent2,self.n_cities), self.crossover)
                    child1, child2 = eval(f"cross_method{self.crossover_call}")
                else:
                    child1, child2 = parent1, parent2

                # Mutate offspring
                if random.random() < self.m_rate:
                    child1 = getattr(mu,self.mutation)(child1)
                if random.random() < self.m_rate:
                    child2 = getattr(mu,self.mutation)(child2)

                offspring.append(child1)
                offspring.append(child2)

            if self.save_results:
                # Compute the metrics
                generation_data = {
                    generation: {
                        "mean_path_distance": np.mean(path_distance),
                        "median_path_distance": np.median(path_distance),
                        "min_path_distance": np.min(path_distance),
                        "max_path_distance": np.max(path_distance)
                    }
                }

                # Append the current generation data
                data.append(generation_data)

            self.population = np.array(offspring)

            if self.print_results:
                # Print progress
                if generation % self.print_rate == 0:
                    logging.info(f"Generation {generation} - Best cromosome: {self.population[int(np.argmin(path_distance))]}, Best path_distance: {np.min(path_distance)}")
                    self.plot_route(self.population[int(np.argmin(path_distance))])
        
        best_chromosome = self.population[int(np.argmin(path_distance))]
        # Get the best chromosome
        if self.print_results:
            logging.info(f"Generation {generation} - Best cromosome: {best_chromosome}, Best path_distance: {np.min(path_distance)}")
        
        # Save the results
        if self.save_results:
            with open(self.results_path, "w") as f:
                json.dump(data, f)

        return best_chromosome , np.min(path_distance)