import numpy as np
import pandas as pd
import streamlit as st

# ================================================================
# Custom GA Problem: 80-bit chromosome, best when 50 ones (fitness=80)
# ================================================================

def fitness_function(x: np.ndarray) -> float:
    ones = np.sum(x)
    return 80 - abs(ones - 50)   # Maximum = 80 when ones = 50


# ================================================================
# GA Operators
# ================================================================

def init_population(pop_size: int, dim: int, rng):
    return rng.integers(0, 2, size=(pop_size, dim), dtype=np.int8)


def tournament_selection(fitness, k, rng):
    idxs = rng.integers(0, len(fitness), size=k)
    return idxs[np.argmax(fitness[idxs])]


def one_point_crossover(a, b, rng):
    point = rng.integers(1, len(a))
    return (
        np.concatenate([a[:point], b[point:]]),
        np.concatenate([b[:point], a[point:]])
    )


def bit_mutation(x, mut_rate, rng):
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def evaluate_population(pop):
    return np.array([fitness_function(ind) for ind in pop], dtype=float)


# ================================================================
# GA Run Function
# ================================================================

def run_ga(pop_size=300, dim=80, generations=50, cx_rate=0.9,
           mut_rate=0.01, tournament_k=3, elitism=2, seed=42):

    rng = np.random.default_rng(seed)

    # Initial population
    pop = init_population(pop_size, dim, rng)
    fitness = evaluate_population(pop)

    history_best, history_avg, history_worst = [], [], []

    for gen in range(generations):

        # Logging
        best_fit = np.max(fitness)
        avg_fit = np.mean(fitness)
        worst_fit = np.min(fitness)

        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        # Elitism
        E = elitism
        elite_idx = np.argpartition(fitness, -E)[-E:]
        elites = pop[elite_idx].copy()

        new_pop = []

        # Generate new population
        while len(new_pop) < pop_size - E:
            p1 = pop[tournament_selection(fitness, tournament_k, rng)]
            p2 = pop[tournament_selection(fitness, tournament_k, rng)]

            # Crossover
            if rng.random() < cx_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = bit_mutation(c1, mut_rate, rng)
            c2 = bit_mutation(c2, mut_rate, rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size - E:
                new_pop.append(c2)

        pop = np.vstack([np.array(new_pop), elites])
        fitness = evaluate_population(pop)

    # Final results
    best_idx = np.argmax(fitness)

    return {
        "best": pop[best_idx],
        "best_fitness": fitness[best_idx],
        "history": pd.DataFrame({
            "Best": history_best,
            "Average": history_avg,
            "Worst": history_worst
        }),
        "final_population": pop,
        "final_fitness": fitness
    }


# ================================================================
# Streamlit UI
# ================================================================

st.title("Genetic Algorithm — Custom Bit Pattern (80 bits, best at 50 ones)")
st.caption("Population = 300, Generations = 50, Fitness max = 80 at exactly 50 ones.")

if st.button("Run Genetic Algorithm", type="primary"):

    result = run_ga()

    # Show chart
    st.subheader("Fitness Progress")
    st.line_chart(result["history"])

    # Show best solution
    st.subheader("Best Individual Found")
    st.write(f"Best Fitness: {result['best_fitness']}")

    bitstring = ''.join(map(str, result['best'].astype(int)))
    st.code(bitstring)

    st.write(f"Number of ones: {np.sum(result['best'])} / 80")

    # Store population for table
    st.session_state["pop"] = result["final_population"]
    st.session_state["fit"] = result["final_fitness"]

# ================================================================
# Final Population Table
# ================================================================

st.subheader("Final Population (First 20)")
if st.button("Show Final Table"):

    if "pop" not in st.session_state:
        st.info("Run the algorithm first.")
    else:
        pop = st.session_state["pop"]
        fit = st.session_state["fit"]

        df = pd.DataFrame(pop[:20])
        df["fitness"] = fit[:20]

        st.dataframe(df, use_container_width=True)
