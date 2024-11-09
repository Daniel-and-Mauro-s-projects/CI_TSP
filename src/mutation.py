import random
import numpy as np

class mutation:

    @staticmethod
    def exchange(path):
        ''' This method picks two random cities and switches them.'''
        i, j = random.sample(range(len(path)), 2)
        # We invert the order of the cities in positions i and j
        path[i], path[j] = path[j], path[i]
        return path

    @staticmethod
    def insertion(path):
        ''' This method picks a random city and moves it to a random position.'''
        i, j = random.sample(range(len(path)), 2)
        # We move city in position i to position j
        path=np.insert(np.delete(path,i), j, path[i]) 
        return path

    @staticmethod
    def IVM(path):
        '''
        Inversion mutation operator.
        The IVM operator inverts the order of a subset of cities in the path and inserts it back randomly.
        '''
        cut = random.randint(0, len(path) - 1)
        p = 1 / (4 * len(path))
        length = np.random.geometric(p)
        
        # Adjust length if it exceeds the path boundaries
        length = min(length, len(path) - cut)
        
        # Extract the subset
        subset = path[cut:cut + length]
        path = np.delete(path,list(range(cut,cut + length)))
        # We reverse the subset
        subset=subset[::-1]

        # Insert the reversed subset back into the path
        new_index = random.randint(0, len(path))  
        path=np.insert(path, new_index, subset)

        return path