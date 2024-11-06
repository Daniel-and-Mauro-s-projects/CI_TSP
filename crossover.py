import random

class Crossover:
    def __init__(self, parent1: list[int], parent2: list[int], number_of_pos: int) -> None:
        self.parent1 = parent1
        self.parent2 = parent2
        self.number_of_pos = number_of_pos

    def OX1(self, first_cut: int = None, second_cut: int = None) -> list[int]:
        '''
        Order crossover operator. 
        The OX1 exploits a property of the path representation, that the order of cities (not their positions) are important.
        
        Args:
            parent1: The first parent chromosome.
            parent2: The second parent chromosome.

        Returns:
            list[int]: The offspring chromosome.
        '''
        # Use the provided cuts if specified, otherwise fall back to instance attributes
        first_cut = first_cut if first_cut is not None else self.first_cut
        second_cut = second_cut if second_cut is not None else self.second_cut
        
        offspring1 = self.parent2[first_cut:second_cut].copy()
        offspring2 = self.parent1[first_cut:second_cut].copy()

        # Fill the rest of the offspring with the remaining cities from the parents
        for city in self.parent1[second_cut:] + self.parent1[:second_cut]:
            if city not in offspring1:
                offspring1.append(city)

        for city in self.parent2[second_cut:] + self.parent2[:second_cut]:
            if city not in offspring2:
                offspring2.append(city)
        
        return offspring1, offspring2

    def POS(self) -> tuple[list[int], list[int]]:
        '''
        Position-based crossover operator.
        The POS operator selects a subset of positions from one parent and fills the remaining positions with the cities from the other parent, maintaining their order and avoiding duplicates.
        '''
        # Select random positions for the subset
        positions = random.sample(range(len(self.parent1)), self.number_of_pos)
        print(positions)
        
        # Initialize offspring with None
        offspring1 = [None] * len(self.parent1)
        offspring2 = [None] * len(self.parent2)
        
        # Copy the selected positions from parent2 to offspring1 and from parent1 to offspring2
        for pos in positions:
            offspring1[pos] = self.parent2[pos]
            offspring2[pos] = self.parent1[pos]

        print(offspring1, offspring2)
        
        # Function to fill the remaining positions
        def fill_offspring(offspring, parent):
            current_index = 0
            for city in parent:
                if city not in offspring:
                    # Find the next available position
                    while offspring[current_index] is not None:
                        current_index += 1
                    assert current_index < len(offspring)
                    offspring[current_index] = city
            return offspring
        
        # Fill the remaining positions for both offspring
        offspring1 = fill_offspring(offspring1, self.parent1)
        offspring2 = fill_offspring(offspring2, self.parent2)
        
        return offspring1, offspring2

# Usage Example
if __name__ == "__main__":
    parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
    parent2 = [2, 4, 6, 8, 7, 5, 3, 1]
    number_of_pos = 3  # Number of positions to copy from each parent
    
    cross = Crossover(parent1, parent2, number_of_pos)
    offspring1, offspring2 = cross.POS()
    
    print("Parent 1:", parent1)
    print("Parent 2:", parent2)
    print("Offspring 1:", offspring1)
    print("Offspring 2:", offspring2)

    offspring1, offspring2 = cross.OX1(2, 5)
    print("Offspring 1 (OX1):", offspring1)
    print("Offspring 2 (OX1):", offspring2)