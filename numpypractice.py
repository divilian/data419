# NumPy practice worksheet (up to +20XP)
# Instructions: fill in the "YOUR CODE GOES HERE" sections by carefully
#   reading and following the instructions.
# If any requested operations are impossible, set the corresponding variable to
#   the exact string "u cant do dat" instead.
# Then run the script to see which items are correct!
import numpy as np

score = 0

# ---------------------------
# Item 1: Matrix multiplication
# Given the matrices x and y below, compute xy and yx using matrix
# multiplication. Put the answers in variables x_mm_y and y_mm_x, respectively.
# (Be sure to read the sentence above that starts with "If".)
x = np.array([[5.0, 3.0],
              [2.0, 2.5],
              [1.0, 9.0]])
y = np.array([[2.0, 2.0],
              [3.0, 4.0]])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 2: Scalar/vector ops & addition
# Compute the values 2v and v+w, for v and w arrays below. Put the answers in
# variables called v2 and v_plus_w.
v = np.array([1., 2., 3.])
w = np.array([4., 5., 6.])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 3: Dot product
# Compute the dot product of v and w, and also the dot product of w and v. Take
# a moment and ask yourself: what should the answers approximately be? Which
# answer should be larger? Put the answers in dot_vw and dot_wv.
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 4: Norms
# Compute the Euclidean norm and the Manhattan norm of the v vector, and store
# those results in variables called v_l2 and v_l1, respectively. Ask yourself:
# which should be larger?
# Then normalize the vector v (using the Euclidean norm) and put the normalized
# result in the variable v_normalized. Ask yourself: what should the Euclidean
# norm of this normalized vector be? Compute it and see if you're right.
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 5: Euclidean distance
# Compute the Euclidean ("crow flies") distance between the points at the
# arrow-ends of vectors v and w and put it in a variable called dist_vw.
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 6: Cosine similarity
# Compute the cosine similarity between the vectors v and w, and store the
# answer in a variable called cos_vw.
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 7: Transpose
# Set a variable called M_T to the transpose of the matrix below. Then set a
# variable called M_T_T to that transpose transposed again. Ask yourself, what
# should M_T_T look like? Print it out and see if you're right.
M = np.array([[1., 2., 3.],
              [4., 5., 6.]])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 8: Hadamard (element-wise) prod and Gram matrices.
# Set a variable A_square_elem to a 2x2 matrix that has each item of A, but
# squared. (Hint: this is super super easy to do in one operation. Don't treat
# each element as its own separate thing, square each, and then reassemble the
# matrix.) Then set variables A_mm_AT and AT_mm_A to be A times its transpose,
# and A-transpose times A, respectively. Ask yourself: should A_mm_AT and
# AT_mm_A have the same value? Print them out and see if you're right.
A = np.array([[1., 2.],
              [3., 4.]])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 9: Row/col sums
# Given the matrix below, set variables row_sums and col_sums to be vectors
# with the sums of each row, and the sum of each column, respectively. Ask
# yourself: what shape/dimension should each of those variables be? Print them
# out and see if you're right.
C = np.array([[1., 2.],
              [3., 4.],
              [5., 6.]])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 10: Indexing & slicing
# Continuing to use the matrix C, from above, set a variable called first_row
# to be a vector that is the first row of C, and one called second_col to be a
# vector holding its second column. (Both operations should be really short and
# sweet, and should not involve you re-creating a vector from scratch with the
# desired contents.)
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 11: Reshape & flatten
# Given the vector "a" below, create a 3x4 matrix called M34 with the same 12
# elements rearranged into three rows. Then, use M34 to create a "flattened"
# version of M34 that has all the elements in a single vector again, and store
# that in a variable called a_flat. (The code should be simple and short.)
a = np.arange(12.0)
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 12: Squeeze & expand_dims
# Examine the array q below, including its shape. Set its shape to a variable
# called q_shape. Then create an array "simple" that has the same information
# but is of shape (2,3,2) (i.e., it removes the unnecessary, 1-dimensional
# axes). Then create an array "complicated" that has the same information but
# is of shape (1,2,1,1,3,2,1) (i.e., it inserts a couple of additional,
# unnecessary, 1-dimensional axes).
q = np.array([[[[[1.], [2.]], [[2.], [3.]], [[3.], [4.]]],
               [[[1.], [2.]], [[2.], [3.]], [[3.], [4.]]]]])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 13: Stack & concatenate
# Given vectors u1 and u2, create a 2x3 matrix called stack_uv that has u1 on
# top of u2. Then create a 6-dimensional vector called cat_uv that has u1's
# contents followed by u2's.
u1 = np.array([1., 2., 3.])
u2 = np.array([4., 5., 6.])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 14: Reductions & means
# Continuing with the C matrix defined above, set three variables sum_C, max_C,
# and mean_C that have scalars with the sum, max, and mean of all C's elements.
# (The code should be simple and short.)
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 15: Batched matrix-mult
# Print out and consider the arrays X and Y below. Ask yourself: what shapes
# are X and Y? Should X @ Y be allowed? Should Y @ X? If so, what is shape of
# each? Put your answers to these in variables called X_dot_Y_shape and
# Y_dot_X_shape.
X = np.stack([C, C + 1], axis=0)
D = np.array([[7., 8., 9., 10.],
              [10., 11., 12., 13.]])
Y = np.stack([D, D], axis=0)
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 16: Scalars (rank-0)
# Compute the sum, and the product, of the two scalar arrays below, then use
# .item() to get the actual float values. Save those float values in variables
# called a_plus_b and a_times_b.
a0 = np.array(3.0)
b0 = np.array(2.0)
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 17: Exponentials
# Make a 1-d array called "values" with the following values: -10, -1, 0,
# .001, 1, e, and 10. (For "e" you can use the constant "np.e") Now, take a
# moment and ask yourself: "approximately what should I expect to get if I take
# e-to-the each one of those values?"
# Then, actually compute "e-to-the" that array, and store it in a variable
# called e_to_the. (In other words, the first element should be
# e-to-the-negative-10, the second element should be e-to-the-negative-1, etc.
# This can be done in one line, with no loop required.) Print it out for
# yourself. Were your guesses right?
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 18: Logs
# You still have your 1-d array called "values" from above.
# Now, take a moment and ask yourself: "approximately what should I expect to
# get if I take the (natural) log of each one of those values? Can I even *do*
# that for all of those values? And if not, which ones can't I do it for?"
# Then, actually compute the log of all values in that array (whether or not
# you think you can do them; NumPy will give NaN for negative values and
# negative infinity for zero), and store it in a variable called logs. (In
# other words, the first element should be the natural log of -10, the second
# element should be the log of -1, etc. This can be done in one line, with no
# loop required.) Print it out for yourself. Were your guesses right?
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 19: Sigmoid function
# You still have your 1-d array called "values" from above.
# Now, take a moment and ask yourself: "approximately what should I expect to
# get if I ran the sigmoid function on each one of those values?"
# Then, actually compute the sigmoid of all values in that array, and store it
# in a variable called sigmoids. (In other words, the first element should be
# the sigmoid function applied to the number -10, etc. I recommend writing a
# function called sigmoid() that takes an array argument and returns an array
# result, then call it to produce your answer with one line of code.)
# Print it out for yourself. Were your guesses right?
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 20: One neuron
#
# Define a function called "jezebel_neuron()" that will compute the probability
# that Jezebel will be attracted to a particular romantic partner. It should
# take a rank-1, 5-dimensional array as input, and return a scalar probability.
# The output should be that of a single neuron (put another way, a single
# logistic regression) with the weights equal to (in order): 5, -1.2, -2, 0,
# 6.
# Then execute this function on the three variables filbert, wendell, and biff,
# defined below. Before doing so, ask yourself: what range would I expect the
# answer to be in? And would I expect the answer to be higher for filbert,
# wendell, or biff? Then print out your answers and see if you're right. Store
# your answers in variables called filbert_prob, wendell_prob, and biff_prob.
# Finally, stack the three boys in a single matrix (to make a rank-2 array of
# shape 3x5) called "boys". Filbert should be on row 1, Wendell on row 2, and
# Biff on row 3. Then run your jezebel_neuron() function on it all in one go,
# and store the answers in a 3-element vector called boy_probs.

filbert = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
wendell = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
biff = np.array([1.0, 0.0, 1.0, 0.0, 0.0])

# === YOUR CODE GOES HERE ===









# =====================================================
# ================== CHECKER SCRIPT ===================
def exists_all(names: list) -> bool:
    g = globals()
    return all(name in g for name in names)


def check_item(idx: int, needed: list, predicate):
    global score
    if not exists_all(needed):
        print(f"(Item #{idx} incomplete.)")
        return
    try:
        ok = predicate()
    except Exception:
        ok = False
    if ok:
        print(f"Item #{idx} correct!")
        score += 1
    else:
        print(f"Item #{idx} INCORRECT!")


def _item1_expected():
    exp_xy = np.array([[19.0000, 22.0000],
                       [11.5000, 14.0000],
                       [29.0000, 38.0000]])
    exp_yx = "u cant do dat"
    return exp_xy, exp_yx


# Run all checks
check_item(1, ["x_mm_y", "y_mm_x"], lambda: (
    isinstance(x_mm_y, np.ndarray)
    and np.allclose(x_mm_y, _item1_expected()[0], atol=1e-4)
    and isinstance(y_mm_x, str)
    and y_mm_x == _item1_expected()[1]
))
check_item(2, ["v2", "v_plus_w"], lambda: (
    np.allclose(v2, np.array([2., 4., 6.]))
    and np.allclose(v_plus_w, np.array([5., 7., 9.]))
))
check_item(3, ["dot_vw", "dot_wv"], lambda: (
    np.isclose(dot_vw, np.dot(v, w))
    and np.isclose(dot_wv, np.dot(w, v))
))
check_item(4, ["v_l2", "v_l1", "v_normalized"], lambda: (
    np.isclose(v_l2, np.linalg.norm(v), atol=1e-5)
    and np.isclose(v_l1, np.linalg.norm(v, 1), atol=1e-5)
    and np.allclose(v_normalized, v / v_l2)
))
check_item(5, ["dist_vw"], lambda: (
    np.isclose(dist_vw, np.linalg.norm(v - w), atol=1e-5)
))
check_item(6, ["cos_vw"], lambda: (
    np.isclose(
        cos_vw,
        np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)),
        atol=1e-6,
    )
))
check_item(7, ["M_T", "M_T_T"], lambda: (
    np.array_equal(M_T, M.T) and np.array_equal(M, M_T_T)
))
check_item(8, ["A_square_elem", "A_mm_AT", "AT_mm_A"], lambda: (
    np.array_equal(A_square_elem, A * A)
    and np.array_equal(A_mm_AT, A @ A.T)
    and np.array_equal(AT_mm_A, A.T @ A)
))
check_item(9, ["row_sums", "col_sums"], lambda: (
    np.array_equal(row_sums, C.sum(axis=1))
    and np.array_equal(col_sums, C.sum(axis=0))
))
check_item(10, ["first_row", "second_col"], lambda: (
    np.array_equal(first_row, C[0])
    and np.array_equal(second_col, C[:, 1])
))
check_item(11, ["M34", "a_flat"], lambda: (
    np.array_equal(M34, a.reshape(3, 4))
    and np.array_equal(a_flat, a.reshape(3, 4).flatten())
))
check_item(12, ["q_shape", "simple", "complicated"], lambda: (
    tuple(q_shape) == (1, 2, 3, 2, 1)
    and simple.shape == (2, 3, 2)
    and complicated.shape == (1, 2, 1, 1, 3, 2, 1)
    and np.array_equal(q.flatten(), simple.flatten())
    and np.array_equal(complicated.flatten(), simple.flatten())
))
check_item(13, ["stack_uv", "cat_uv"], lambda: (
    np.array_equal(stack_uv, np.stack([u1, u2], axis=0))
    and np.array_equal(cat_uv, np.concatenate([u1, u2], axis=0))
))
check_item(14, ["sum_C", "mean_C", "max_C"], lambda: (
    np.isclose(sum_C, C.sum())
    and np.isclose(max_C, C.max())
    and np.isclose(mean_C, C.mean())
))
check_item(15, ["X_dot_Y_shape", "Y_dot_X_shape"], lambda: (
    tuple(X_dot_Y_shape) == (2, 3, 4)
    and isinstance(Y_dot_X_shape, str)
    and Y_dot_X_shape == "u cant do dat"
))
check_item(16, ["a_plus_b", "a_times_b"], lambda: (
    isinstance(a_plus_b, float)
    and a_plus_b == float(a0 + b0)
    and isinstance(a_times_b, float)
    and a_times_b == float(a0 * b0)
))
check_item(17, ["e_to_the"], lambda: (
    np.allclose(
        e_to_the,
        np.array([
            0.0000453999, 0.3678794412, 1.0, 1.0010005002,
            2.7182818285, 15.1542622415, 22026.4657948067,
        ]),
        atol=1e-4,
    )
))
check_item(18, ["logs"], lambda: (
    np.isnan(logs[0])
    and np.isnan(logs[1])
    and np.isneginf(logs[2])
    and np.allclose(
        logs[3:],
        np.array([-6.90775527898, 0.0, 1.0, 2.30258509299]),
        atol=1e-4,
    )
))
check_item(19, ["sigmoids"], lambda: (
    np.allclose(
        sigmoids,
        np.array([
            0.0000453979, 0.2689414214, 0.5, 0.50025,
            0.7310585786, 0.9380968326, 0.9999546021,
        ]),
        atol=1e-4,
    )
))
check_item(20, ["boy_probs", "filbert_prob", "wendell_prob", "biff_prob"],
           lambda: (
    np.isclose(filbert_prob, 0.9999832986, atol=1e-4)
    and np.isclose(wendell_prob, 0.0391657228, atol=1e-4)
    and np.isclose(biff_prob, 0.9525741268, atol=1e-4)
    and np.allclose(
        boy_probs,
        np.array([0.9999832986, 0.0391657228, 0.9525741268]),
        atol=1e-4,
    )
))

print(f"You got +{score}XP! (out of a possible 20XP)")
