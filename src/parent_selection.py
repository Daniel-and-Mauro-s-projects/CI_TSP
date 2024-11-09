
import numpy as np
import random

class parent_selection:
    """Methods for parent selection."""

    # Here the distance is not a fitness function, it is sort of a cost function (the lower the better)  
    @staticmethod
    def roulette_selection(population: list[list[int]], distance_scores: list[float]) -> list[int]:
        """
        Selects a chromosome based on a weighted probability.

        Args:
            population (list[list[int]]): The population to select from.
            distance_scores (list[float]): The distance scores of the population.

        Returns:
            list[int]: The selected chromosome.
        """
        # The lower the distance score, the better. In order to enforce this, we use the inverse of the distance score as the weight.
        weights=np.array([1 / x for x in distance_scores])
        return random.choices(population, weights=weights/np.sum(weights), k=1)[0]
    
    @staticmethod
    def rank_roulette_selection(population: list[list[int]], distance_scores: list[float]) -> list[int]:
        """
        Selects a chromosome based on a weighted probability using the rank of the chromosome.

        Args:
            population (list[list[int]]): The population to select from.
            distance_scores (list[float]): The distance scores of the population.

        Returns:
            list[int]: The selected chromosome.
        """
        # We get a sorted list of indices of the population based on the distance scores (from worst to best)
        ranks = np.argsort(distance_scores)

        # The better ranks have a higher probability of being selected
        weights = np.array([1 / (i + 1) for i in ranks])
        return random.choices(population, weights=weights, k=1)[0]
    
    @staticmethod
    def random_selection(population: list[list[int]],distance_scores: list[float]=None) -> list[int]:
        """Selects a random chromosome from the population."""
        return random.choice(population)
    
    @staticmethod
    def tournament_selection(population: list[list[int]], distance_scores: list[float], tournament_size: int = 3) -> list[int]:
        """
        Selects the best chromosome from a random subset of the population.

        Args:
            population (list[list[int]]): The population to select from.
            tournament_size (int): The size of the subset to select from.

        Returns:
            list[int]: The selected chromosome.
        """
        selected_index = random.choices(range(len(population)), k=tournament_size)
        # Sort the selected chromosomes by their distance score (from worst to best)
        selected = sorted(selected_index, key=lambda x: distance_scores[x], reverse=False)
        return population[selected[0]]