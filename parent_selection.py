
import numpy as np
import random

class parent_selection:
    """Methods for parent selection."""

    @staticmethod
    def tournament_selection(population: list[list[int]], fitness_scores: list[float], tournament_size: int = 3) -> list[int]:
        """
        Selects the best chromosome from a random subset of the population.

        Args:
            population (list[list[int]]): The population to select from.
            tournament_size (int): The size of the subset to select from.

        Returns:
            list[int]: The selected chromosome.
        """
        selected_index = random.choices(range(len(population)), k=tournament_size)
        selected = sorted(selected_index, key=lambda x: fitness_scores[x])
        return population[selected[0]]
    
    @staticmethod
    def roulette_selection(population: list[list[int]], fitness_scores: list[float]) -> list[int]:
        """
        Selects a chromosome based on a weighted probability.

        Args:
            population (list[list[int]]): The population to select from.
            fitness_scores (list[float]): The fitness scores of the population.

        Returns:
            list[int]: The selected chromosome.
        """
        return random.choices(population, weights=fitness_scores, k=1)[0]