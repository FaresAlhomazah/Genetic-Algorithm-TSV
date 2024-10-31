import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgorithmTSP:
    def __init__(self, n_cities, population_size=100, generations=100, mutation_rate=0.01, width=100, height=100):
        self.n_cities = n_cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.width = width
        self.height = height
        self.city_names, self.cities = self.generate_cities(n_cities)
        self.best_route = None                                     
        self.best_distance = float('inf')

    def generate_cities(self, n_cities):
        city_names = [f"City {i+1}" for i in range(n_cities)]
        cities = np.random.rand(n_cities, 2) * [self.width, self.height]
        print(f"generate_cities : {city_names," fares ", cities}")
        return city_names, cities
    
    def calculate_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            city1, city2 = self.cities[route[i]], self.cities[route[i + 1]]
            distance += np.linalg.norm(city2 - city1)
        distance += np.linalg.norm(self.cities[route[-1]] - self.cities[route[0]]) 
        return distance
    
    def create_initial_population(self):
        return [self.discrete_valued_representation() for _ in range(self.population_size)] 
    
    def discrete_valued_representation(self):
        return random.sample(range(self.n_cities), self.n_cities)
    
    def evaluate_fitness(self, population):
        fitness = []
        for route in population:
            distance = self.calculate_distance(route)
            if distance > 0:  # Avoid division by zero
                fitness.append(1 / distance)
            else:
                fitness.append(0)  
        return fitness
        
    def roulette_wheel_selection(self, population, fitness):
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        parents = random.choices(population, probabilities,k=2)
        return parents

    def rank_based_selection(self, population, fitness):
        sorted_pop = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)
        ranks = [i + 1 for i in range(len(sorted_pop))]
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        parents = random.choices([p for _, p in sorted_pop], probabilities, k=2)
        return parents

    def tournament_selection(self, population, fitness, tournament_size=10):
        tournament = random.sample(list(zip(fitness, population)), tournament_size)
        winners = sorted(tournament, key=lambda x: x[0], reverse=True)[:2]
        return winners[0][1], winners[1][1]


    def random_selection(self, population):
        return random.sample(population, 2)

    def select_parents(self, population, fitness, selection_type='roulette'):
        if selection_type == '1':
            return self.rank_based_selection(population, fitness)
        elif selection_type == '2':
            return self.tournament_selection(population, fitness)
        elif selection_type == '3':
            return self.random_selection(population)
        else:  # Default to roulette wheel
            return self.roulette_wheel_selection(population, fitness)

    def one_point_crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + [x for x in parent2 if x not in parent1[:point]]
        child2 = parent2[:point] + [x for x in parent1 if x not in parent2[:point]]
        return child1, child2

    def two_point_crossover(self, parent1, parent2):
        point1 = random.randint(1, len(parent1) - 1)
        point2 = random.randint(1, len(parent1) - 1)
        if point1 > point2:
            point1, point2 = point2, point1
        child1 = parent1[:point1] + [x for x in parent2 if x not in parent1[:point1]] + parent1[point2:]
        child2 = parent2[:point1] + [x for x in parent1 if x not in parent2[:point1]] + parent2[point2:]
        return child1, child2

    def uniform_crossover(self, parent1, parent2):
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2

    def crossover(self, parent1, parent2, crossover_type='one_point'):
        if crossover_type == '1':
            return self.one_point_crossover(parent1, parent2)
        elif crossover_type == '2':
            return self.two_point_crossover(parent1, parent2)
        elif crossover_type == '3':
            return self.uniform_crossover(parent1, parent2)
        else:
            raise ValueError("Crossover type not recognized")

    def mutate(self, route):
        for i in range(len(route)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(route) - 1)
                route[i], route[j] = route[j], route[i]
        return route

    def replace_parents(self, population, offspring):
        return offspring  

    def replace_randomly(self, population, offspring):
        return random.sample(population, len(population) - len(offspring)) + offspring

    def replace_worst(self, population, offspring, fitness):
        ranked_population = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)
        population = [p for _, p in ranked_population[:len(population) - len(offspring)]]
        population.extend(offspring)
        return population

    def replace_population(self, population, offspring, fitness, replacement_type='worst'):
        if replacement_type == '1':
            return self.replace_parents(population, offspring)
        elif replacement_type == '2':
            return self.replace_randomly(population, offspring)
        else:  
            return self.replace_worst(population, offspring, fitness)

    def run(self, selection_type='roulette', crossover_type='one_point', replacement_type='worst'):
        population = self.create_initial_population()

        for generation in range(self.generations):
            fitness = self.evaluate_fitness(population)

            best_gen_route = population[np.argmax(fitness)]
            best_gen_distance = self.calculate_distance(best_gen_route)

            if best_gen_distance < self.best_distance:
                self.best_route = best_gen_route
                self.best_distance = best_gen_distance

            offspring = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(population, fitness, selection_type)
                child1, child2 = self.crossover(parent1, parent2, crossover_type)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])

            population = self.replace_population(population, offspring, fitness, replacement_type)

        return self.best_route, self.best_distance

    def plot_route(self):
        route_cities = np.array([self.cities[i] for i in self.best_route] + [self.cities[self.best_route[0]]])
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(route_cities[:, 0], route_cities[:, 1], 'o-', label="Path", markersize=8, linewidth=2)

        for i, city in enumerate(self.best_route):
            plt.text(self.cities[city][0], self.cities[city][1], self.city_names[city], fontsize=12, ha='right')

            if i < len(self.best_route) - 1:
                city1, city2 = route_cities[i], route_cities[i + 1]
                distance = np.linalg.norm(city2 - city1)
                mid_point = (city1 + city2) / 2
                plt.text(mid_point[0], mid_point[1], f'{distance:.2f}', fontsize=10, color='blue')
        
        start_city = self.cities[self.best_route[0]]
        plt.scatter(start_city[0], start_city[1], c='green', label="Start", s=200, edgecolors='black', zorder=5)

        end_city = self.cities[self.best_route[-1]]
        if np.array_equal(start_city, end_city):
            plt.scatter(end_city[0], end_city[1], c='red', label="End/Return", s=200, edgecolors='black', zorder=5)
        else:
            plt.scatter(end_city[0], end_city[1], c='red', label="End", s=200, edgecolors='black', zorder=5)

        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', label="Cities", s=100)
        
        plt.title(f'Best Route (Total Distance: {self.best_distance:.2f} km)', fontsize=16)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    try:
        n_cities_input = input("Enter the number of cities (default 10): ").strip()
        n_cities = int(n_cities_input) if n_cities_input else 10 

        mutation_rate_input = input("Enter mutation rate (e.g., 0.01, default 0.01): ").strip()
        mutation_rate = float(mutation_rate_input) if mutation_rate_input else 0.01 

        selection_method = input("Select selection method (rank (1), tournament (2), random (3), roulette (4), default 1): ").strip()
        selection_method = selection_method if selection_method else '1'

        crossover_method = input("Select crossover method (one-point (1), two-point (2), uniform (3), default 1): ").strip()
        crossover_method = crossover_method if crossover_method else '1'  

        replacement_method = input("Select replacement method (parents (1), random (2), worst (3), default 1): ").strip()
        replacement_method = replacement_method if replacement_method else '1' 

        ga = GeneticAlgorithmTSP(n_cities=n_cities, population_size=100, generations=500, mutation_rate=mutation_rate)

        best_route, best_distance = ga.run(
            selection_type=selection_method,          
            crossover_type=crossover_method,         
            replacement_type=replacement_method
        )

        print(f"Best route: {best_route}")
        print(f"Best distance: {best_distance:.2f} km")

        ga.plot_route()

    except ValueError as e:
        print("Invalid input. Please enter valid numbers for cities and mutation rate.")
    except Exception as e:
        print(f"An error occurred: {e}")
