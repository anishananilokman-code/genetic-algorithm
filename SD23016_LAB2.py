# YourStudentID_Lab2.py
import random
import numpy as np
import pandas as pd
import streamlit as st

# =====================================
# Fixed GA Requirements (from Lab 2)
# =====================================
POPULATION = 300
CHROMOSOME_LENGTH = 80
TARGET_ONES = 50
BEST_FITNESS_VALUE = 80
GENERATIONS = 50

# GA hyperparameters
TOUR_SIZE = 3
CROSS_RATE = 0.9
MUT_RATE = 1 / CHROMOSOME_LENGTH


# =====================================
# GA Core Functions
# =====================================

def fitness(individual):
    """
    Fitness = 80 when number of ones = 50.
    Drops by 1 for each distance away from 50.
    """
    ones = int(np.sum(individual))
    return BEST_FITNESS_VALUE - abs(ones - TARGET_ONES)


def create_population(size, length):
    """Generate initial random 0/1 population."""
    return np.random.randint(0, 2, size=(size, length), dtype=np.int8)


def selection(pop, fit):
    """Tournament selection."""
    candidates = np.random.randint(0, len(pop), size=TOUR_SIZE)
    winner = candidates[np.argmax(fit[candidates])]
    return pop[winner].copy()


def crossover(p1, p2):
    """Single-point crossover."""
    if np.random.rand() > CROSS_RATE:
        return p1.copy(), p2.copy()

    point = np.random.randint(1, CHROMOSOME_LENGTH)
    child1 = np.concatenate([p1[:point], p2[point:]])
    child2 = np.concatenate([p2[:point], p1[point:]])
    return child1, child2


def mutation(individual):
    """Bit-flip mutation."""
    mask = np.random.rand(CHROMOSOME_LENGTH) < MUT_RATE
    offspring = individual.copy()
    offspring[mask] ^= 1
    return offspring


def evolve(pop):
    """Main GA evolutionary loop."""
    best_curve = []
    global_best = None
    global_best_fit = -999

    for _ in range(GENERATIONS):
        fitness_values = np.array([fitness(ind) for ind in pop])

        # Track best of this generation
        idx = np.argmax(fitness_values)
        gen_best = pop[idx]
        gen_best_fit = fitness_values[idx]
        best_curve.append(gen_best_fit)

        # Update global best
        if gen_best_fit > global_best_fit:
            global_best_fit = gen_best_fit
            global_best = gen_best.copy()

        # Produce next generation
        new_pop = []
        while len(new_pop) < len(pop):
            p1 = selection(pop, fitness_values)
            p2 = selection(pop, fitness_values)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutation(c1))
            if len(new_pop) < len(pop):
                new_pop.append(mutation(c2))

        pop = np.array(new_pop, dtype=np.int8)

    return global_best, global_best_fit, best_curve


# =====================================
# Streamlit Interface
# =====================================
st.set_page_config(page_title="Lab 2 Genetic Algorithm", page_icon="🧬")

st.title("🧬 Genetic Algorithm — 80-Bit Target Pattern")
st.write("""
This GA attempts to evolve an 80-bit string that contains exactly 50 ones.
Fitness is highest (**80**) when the bitstring has 50 ones.
""")

seed = st.number_input("Random seed", value=42)
start_btn = st.button("Run GA", type="primary")

if start_btn:
    random.seed(seed)
    np.random.seed(seed)

    population = create_population(POPULATION, CHROMOSOME_LENGTH)
    best_individual, best_fit, curve = evolve(population)

    ones = int(np.sum(best_individual))
    zeros = CHROMOSOME_LENGTH - ones
    bit_str = ''.join(map(str, best_individual.tolist()))

    st.subheader("🏆 Best Individual Found")
    st.metric("Best Fitness", best_fit)
    st.write(f"Ones: {ones} | Zeros: **{zeros}**")
    st.code(bit_str)

    st.subheader("📈 Fitness Convergence Curve")
    df = pd.DataFrame({"Best Fitness": curve})
    st.line_chart(df)

    if ones == TARGET_ONES:
        st.success("Perfect result achieved! 50 ones → Fitness = 80")
    else:
        st.info("Not exact — but close. Adjust the seed and try again.")