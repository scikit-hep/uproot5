# import uproot
# import uproot.model


# f = uproot.open("./example-objects.root")
# g = f["tgraph"]

# f.close()

# import uproot
# import uproot.writing

# import numpy as np

# # Define x and y coordinates
# x = np.array([1, 2, 3, 4, 5], dtype='float64')
# y = np.array([10, 20, 30, 40, 50], dtype='float64')

# # Define the name and title of the TGraph
# graph_name = "example_graph"
# graph_title = "Example TGraph"

# # Create the TGraph object
# graph = uproot.writing.new_th1x(graph_name, graph_title, x, y)

# # Write the TGraph to a ROOT file
# with uproot.recreate("example.root") as file:
#     file[graph_name] = graph

import uproot

import numpy as np
import pandas as pd

# Define x and y coordinates
x = np.array([1, 2, 3, 4, 5, 7,9,20], dtype='float64')
y = np.array([1, 2, 3, 4, 5, 10, 20, 50], dtype='float64')
errors_x = np.array([0,0,0,0,0],dtype='float64')
errors_y = np.array([0,0,0,0,0],dtype='float64')


d = {'x': x,'y': y, 'errors':errors_x}
d2 = (x,y)

f = uproot.recreate("example.root")

f["mytgraph"] = uproot.as_TGraph(x, y)

f.close()


with uproot.open("example.root") as f:
    g = f["mytgraph"]
    print(g)
    print(g.values())


