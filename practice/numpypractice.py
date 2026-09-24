#!/usr/bin/env python3
'''
DATA 419 -- Assignment #0.5 support file
Stephen Davies, University of Mary Washington, fall 2026
'''

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
# Item 6: More indexing & slicing
# Given the matrix B below, use indexing and slicing (not np.array()) to create
# the following variables:
#   first_four_rows: the first four rows of B
#   last_three_cols: the last three columns of B
#   upper_right: the first four rows and columns 3 through the end
#   every_other_row: rows 1, 3, and 5 (i.e., every other row, starting with
#       the first)
B = np.arange(30).reshape(6, 5)
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
# Item 15: Shapes
# Examine the arrays below. Without re-creating any arrays, set the variables
# v_shape, M_shape, and cube_shape equal to their shapes. Then set rows_M and
# cols_M equal to the number of rows and columns in shape_M, respectively,
# using shape_M.shape.
shape_v = np.array([10., 20., 30., 40.])
shape_M = np.arange(15).reshape(3, 5)
cube = np.arange(24).reshape(2, 3, 4)
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
# Item 19: Random numbers
# Use NumPy's random-number functions to create the following:
#   random_ints: 10 random integers from 1 through 6, inclusive
#   random_choices: 8 random choices from the array ["red", "green", "blue"]
#   random_uniforms: 7 random real numbers uniformly distributed from -2 to 2
#   random_normals: 9 random real numbers from a normal distribution with mean
#       100 and standard deviation 15
#
# Important: use np.random.randint(), np.random.choice(), np.random.uniform(),
# and np.random.normal(), respectively. The checker will test the arrays'
# shapes and whether their values satisfy the requested conditions.
np.random.seed(123)
# === YOUR CODE GOES HERE ===
# ---------------------------
# Item 20: Slicing challenge
# Given the matrix E below, use indexing and slicing (not np.array()) to create
# each requested result:
#   middle: rows 2 through 4 and columns 2 through 4
#   bottom_left: the last three rows and first two columns
#   reverse_rows: all of E, but with the rows in reverse order
#   checkerboard: every other row and every other column, starting with the
#       first row and first column
E = np.arange(1, 31).reshape(5, 6)
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
check_item(6, [
    "first_four_rows", "last_three_cols", "upper_right", "every_other_row"
], lambda: (
    np.array_equal(first_four_rows, B[:4])
    and np.array_equal(last_three_cols, B[:, 2:])
    and np.array_equal(upper_right, B[:4, 2:])
    and np.array_equal(every_other_row, B[::2])
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
check_item(15, [
    "v_shape", "M_shape", "cube_shape", "rows_M", "cols_M"
], lambda: (
    tuple(v_shape) == (4,)
    and tuple(M_shape) == (3, 5)
    and tuple(cube_shape) == (2, 3, 4)
    and rows_M == 3
    and cols_M == 5
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
check_item(19, [
    "random_ints", "random_choices", "random_uniforms", "random_normals"
], lambda: (
    np.array_equal(
        random_ints,
        np.array([6, 3, 5, 3, 2, 4, 3, 4, 2, 2])
    )
    and np.array_equal(
        random_choices,
        np.array(['red', 'green', 'blue', 'green', 'red', 'blue', 'red', 'green'])
    )
    and np.allclose(
        random_uniforms,
        np.array([-0.24571102128150235, -1.7612884135617266, -0.40782297867827433, 0.9519816229281428, -1.270033078186, -1.2981929754100299, 0.12620549536735348])
    )
    and np.allclose(
        random_normals,
        np.array([133.11141152777492, 107.84113703994169, 106.98467133693508, 110.87372838146901, 122.43739788260763, 111.19870887869848, 83.48522117482032, 78.84548194036388, 88.78523024864701])
    )
))
check_item(20, [
    "middle", "bottom_left", "reverse_rows", "checkerboard"
], lambda: (
    np.array_equal(middle, E[1:4, 1:4])
    and np.array_equal(bottom_left, E[-3:, :2])
    and np.array_equal(reverse_rows, E[::-1])
    and np.array_equal(checkerboard, E[::2, ::2])
))

print(f"You got +{score}XP! (out of a possible 20XP)")
