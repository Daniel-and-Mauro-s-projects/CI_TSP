class Crossover:
    def __init__(self, parent1: list[int], parent2: list[int]) -> None:
        self.parent1 = parent1
        self.parent2 = parent2
        self.first_cut = len(parent1) // 3
        self.second_cut = 2 * len(parent1) // 3
    
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


# Usage
cross = Crossover([1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 6, 8, 7, 5, 3, 1])
print(cross.OX1())

# Or, you can specify the cuts
cross = Crossover([1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 6, 8, 7, 5, 3, 1])
print(cross.OX1(2, 6))

