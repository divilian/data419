
# Demonstrate how to "project" a point onto a line. (For dimensionality
# reduction.)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

N = 16

def proj_onto(vec,onto_vec):
    return (vec @ onto_vec) / (onto_vec @ onto_vec) * onto_vec

to_project_onto = np.random.uniform(-5,5,size=2)
plt.figure()
fig, ax = plt.subplots()
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.axline((0,0),to_project_onto,color="red",linewidth=2)
for y in range(-5,6):
    ax.axhline(y=y,color="gray",linewidth=1,linestyle="dotted")
for x in range(-5,6):
    ax.axvline(x=x,color="gray",linewidth=1,linestyle="dotted")
ax.axvline(x=0,color="gray",linewidth=2,linestyle="solid")
ax.axhline(y=0,color="gray",linewidth=2,linestyle="solid")
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)

pts = np.random.uniform(-5,5,size=(N,2))

for pt in pts:
    ax.scatter(pt[0],pt[1],color="blue",marker='o')
    projected_pt = proj_onto(pt, to_project_onto)
    ax.scatter(projected_pt[0],projected_pt[1],color="red",edgecolors="black",
        s=30)
    # (Weirdly, to plot line segments you have to give both x values and then
    # both y values (each in a tuple) instead of one point (x&y) and then the
    # other.)
    ax.plot((pt[0],projected_pt[0]),(pt[1],projected_pt[1]),
        color="gray",linestyle="dashed")

# Show square plot, not rectangle.
fig.show()
