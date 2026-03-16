import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Create a graph
G = nx.Graph()     # Undirected graph
# G = nx.DiGraph() # Directed graph

# Step 2: Add nodes
G.add_node(1)
G.add_nodes_from([2, 3, 4])

# Step 3: Add edges
G.add_edge(1, 2)
G.add_edges_from([(2, 3), (3, 4), (4, 1)])

G.add_edge(1, 2, weight=5)
G.add_edge(2, 3, weight=2)

print(G[1][2]["weight"])

# Step 4: Print graph info
print("Nodes:", G.nodes())
print("Edges:", G.edges())

# Step 5: Basic graph properties
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Step 6: Draw the graph
nx.draw(G, with_labels=True)

plt.show()