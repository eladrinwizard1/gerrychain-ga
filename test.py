import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
import pandas
import tqdm
import random
import ga_psc_districts

graph = Graph.from_file("Counties_Georgia.zip", ignore_errors=True)

# Configure our updaters (everything we want to compute
# for each plan in the ensemble).

# Population updater, for computing how close to equality the district
# populations are. "totpop10" is the population column from our shapefile.
my_updaters = {"population": updaters.Tally("totpop10", alias="population")}

# Use the map at http://www.psc.state.ga.us/pscinfo/districts/pscdistricts.htm
# as a starting assignment

# GeographicPartition comes with built-in ``area`` and ``perimeter`` updaters.
initial_partition = GeographicPartition(graph, assignment=ga_psc_districts.assignment, updaters=my_updaters)

# The recom proposal needs to know the ideal population for the districts so
# that we can improve speed by bailing early on unbalanced partitions.

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

# We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
# of the recom proposal.
proposal = partial(recom,
                   pop_col="TOTPOP",
                   pop_target=ideal_population,
                   epsilon=0.02,
                   node_repeats=2
                  )

# To keep districts about as compact as the original plan, we bound the number
# of cut edges at 2 times the number of cut edges in the initial plan.

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)


# Configure the MarkovChain.

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        # District populations must stay within 2% of equality
        constraints.within_percent_of_ideal_population(initial_partition, 0.02),
        compactness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=1000
)

for partition in chain:
    # for key in partition.keys():
    #     print(key)
    #     print(partition[key])
    print(type(partition))
    print(partition.keys())
    nodes = partition.graph.nodes
    print(nodes[0])
    # data = {node: nodes[node][attribute] for node in nodes}
    for node, part in partition.assignment.items():
        print(node, part)
        print(type(node), type(part))
        break
    print(partition["PRES16"])
    break

# Run the chain, putting the sorted Democratic vote percentages
# into a pandas DataFrame for analysis and plotting.

# This will take about 10 minutes.

# data = pandas.DataFrame(
#     sorted(partition["SEN16"].percents("Democratic"))
#     for partition in chain
# )

# If you install the ``tqdm`` package, you can see a progress bar
# as the chain runs by running this code instead:

data = pandas.DataFrame(
    sorted(partition["SEN16"].percents("Democratic"))
    for partition in chain.with_progress_bar()
)

fig, ax = plt.subplots(figsize=(8, 6))

# Draw 50% line
ax.axhline(0.5, color="#cccccc")

# Draw boxplot
data.boxplot(ax=ax, positions=range(0, len(data.columns)))

# Draw initial plan's Democratic vote %s (.iloc[0] gives the first row)
plt.plot(data.iloc[0], "ro")

# Annotate
ax.set_title("Comparing the current plan to an ensemble")
ax.set_ylabel("Democratic vote % (Senate 2016)")
ax.set_xlabel("Sorted districts")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

plt.show()

# stuff for anno's PA plans comparisons

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

enacted = []
court = []
gov = []
joint = []

ax.set_title("Comparing the ranked vote percentages of the four plans")
ax.set_ylabel("Democratic vote percentage 2012-2018")
ax.set_xlabel("Sorted districts")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

plt.show()
