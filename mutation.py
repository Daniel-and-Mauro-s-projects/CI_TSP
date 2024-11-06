import random

class mutation:

    @staticmethod
    def inversion(x):
        """ This method picks two random cities and switches them."""
        i, j = random.sample(range(len(x)), 2)
        # We invert the order of the cities in positions i and j
        x[i], x[j] = x[j], x[i]
        return x
    
    @staticmethod
    def insertion(x):
        """ This method picks a random city and moves it to a random position."""
        i, j = random.sample(range(len(x)), 2)
        # We move city in position i to position j
        x.insert(j, x.pop(i)) # pop(i) removes the element at position i and returns it
        return x