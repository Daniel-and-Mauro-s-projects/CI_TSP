import random
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

# Type alias for integer NumPy arrays
IntArray = NDArray[np.int_]

class crossover:
    def __init__(self, parent1: IntArray, parent2: IntArray, number_of_pos: int) -> None:
        if len(parent1) != len(parent2):
            raise ValueError("Both parents must have the same length.")
        if number_of_pos > len(parent1):
            raise ValueError("number_of_pos cannot exceed the length of the parents.")
        self.parent1: IntArray = parent1
        self.parent2: IntArray = parent2
        self.number_of_pos: int = number_of_pos
        self.first_cut: int = parent1.shape[0] // 3
        self.second_cut: int = 2 * parent1.shape[0] // 3

    def POS(self) -> Tuple[IntArray, IntArray]:
        '''
        Position-based crossover operator.
        The POS operator selects a subset of positions from one parent and fills the remaining positions with the 
        cities from the other parent, maintaining their order and avoiding duplicates.
        
        Returns:
            Tuple containing:
                - Offspring1 as a NumPy array
                - Offspring2 as a NumPy array
        '''
        # Select random positions for the subset
        positions = random.sample(range(len(self.parent1)), self.number_of_pos)
        # Debugging: print("Selected positions for POS:", positions)
        
        # Initialize offspring with None
        offspring1: List[int] = [None] * len(self.parent1)
        offspring2: List[int] = [None] * len(self.parent2)
        
        # Copy the selected positions from parent2 to offspring1 and from parent1 to offspring2
        for pos in positions:
            offspring1[pos] = int(self.parent2[pos])  # Convert to int for consistency
            offspring2[pos] = int(self.parent1[pos])  # Convert to int for consistency
        
        # Function to fill the remaining positions
        def fill_offspring(offspring: List[int], parent: IntArray) -> List[int]:
            current_index = 0
            for city in parent:
                if city not in offspring:
                    # Find the next available position
                    while current_index < len(offspring) and offspring[current_index] is not None:
                        current_index += 1
                    if current_index < len(offspring):
                        offspring[current_index] = int(city)  # Convert to int for consistency
            return offspring
        
        # Fill the remaining positions for both offspring
        offspring1 = fill_offspring(offspring1, self.parent1)
        offspring2 = fill_offspring(offspring2, self.parent2)
        
        # Convert offspring lists to NumPy arrays
        offspring1_array: IntArray = np.array(offspring1, dtype=int)
        offspring2_array: IntArray = np.array(offspring2, dtype=int)
        
        return offspring1_array, offspring2_array

    def OX1(self, first_cut: int = None, second_cut: int = None) -> Tuple[IntArray, IntArray]:
        '''
        Order Crossover (OX1) operator.
        The OX1 operator selects a subset from one parent and preserves the relative order of the remaining cities from the other parent.
        
        Args:
            first_cut (int, optional): The starting index for the crossover. If None, it uses the default value
            second_cut (int, optional): The ending index for the crossover. If None, it uses the default value
        
        Returns:
            Tuple containing:
                - Offspring1 as a NumPy array
                - Offspring2 as a NumPy array
        '''
        size = len(self.parent1)
        
        if first_cut is None or second_cut is None:
            first_cut, second_cut = self.first_cut, self.second_cut
        
        # Initialize offspring with None
        offspring1: List[int] = [None] * size
        offspring2: List[int] = [None] * size
        
        # Copy the subset from Parent1 to Offspring1 and from Parent2 to Offspring2
        offspring1[first_cut:second_cut] = self.parent1[first_cut:second_cut].tolist()
        offspring2[first_cut:second_cut] = self.parent2[first_cut:second_cut].tolist()
        
        # Function to fill the remaining positions maintaining order
        def fill_offspring_order(offspring: List[int], parent: IntArray) -> IntArray:
            current_index = 0
            for city in parent:
                if city not in offspring:
                    while offspring[current_index] is not None:
                        current_index  += 1
                    offspring[current_index] = int(city)
            # print(offspring, type(offspring))
            return np.array(offspring, dtype=int)
        
        # print(offspring1, type(offspring1))
        # Fill the remaining positions
        offspring1 = fill_offspring_order(offspring1, self.parent2)
        offspring2 = fill_offspring_order(offspring2, self.parent1)
        
        return offspring1, offspring2

# Usage Example
if __name__ == "__main__":
    # Example parents as NumPy integer arrays
    parent1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
    parent2 = np.array([2, 4, 6, 8, 7, 5, 3, 1], dtype=int)
    number_of_pos = 3  # Number of positions to copy from each parent
    
    crossover_operator = Crossover(parent1, parent2, number_of_pos)
    
    # Using POS
    offspring1_pos, offspring2_pos = crossover_operator.POS()
    print("POS Offspring 1 (NumPy Array):", offspring1_pos)
    print("POS Offspring 2 (NumPy Array):", offspring2_pos)
    
    # Using OX1
    offspring1_ox1, offspring2_ox1 = crossover_operator.OX1()
    print("\nOX1 Offspring 1 (NumPy Array):", offspring1_ox1)
    print("OX1 Offspring 2 (NumPy Array):", offspring2_ox1)