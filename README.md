# CI_TSP
Genetic Algorithms+TSP

\textbf{For better explanation of both the repository and the project see Report.pdf}

One of our main goals was to create a flexible and modular implementation that allows for easy experimentation with different configurations and operators. To achieve this, we used Python as the main programming language, and we implemented the parent-selection methods, crossover operators, and mutation operators as separate modules. This design allows us to easily swap out different operators and configurations without changing the core logic of the genetic algorithm. Aside from this, the class that contains the genetic algorithm is also modular, allowing for easy extension and modification. All of these files can be found in the src folder of the project.

To use all of these modules, we set up a Jupyter notebook file tsp.ipynb that contains the data reading function, the grid search function, and most graph plotting cells. In this notebook, we also wrote a Usage Example section, where we show how to use the genetic algorithm class and all of its parameters. Although if more insight is needed, all the functions and classes are documented in their respective docstrings.

# Usage example 

distances, cities= read_data('bays29.tsp')

## Instantiate the genetic algorithm
genetic= TSP_Genetic(generations=100,
                    print_results=True,
                    print_rate=10,
                    m_rate=0.05,
                    c_rate=0.8,
                    select_parents="tournament_selection",
                    tournament_size=5,
                    crossover="OX1",
                    crossover_call="()", 
                    mutation="insertion",
                    elitism=3,
                    results_path= "results/distance_statistics.json",
                    save_results=True)

## Create the population:
population = create_population(200, cities)

## Run the genetic algorithm
final_population = genetic.run(population,cities,distances)