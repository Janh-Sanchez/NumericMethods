import math

# This is the math function we're gonna use
def function(x: float):
    return round(pow(math.e, 3 * x) - 4,4)

# Logic of the method
def midpoint(a: float, b: float):
    return (a + b)/2

def have_opposite_signs(a: float, b: float):
    return a * b < 0

interval_l = int(input("Enter the lower interval: "))
interval_u = int(input("Enter the upper interval: "))
i_error = int(input("Enter the error: "))

# This is the form to calcule the error in the iter
error = (interval_u - interval_l)/2

i = 1
print("Interval lower:", interval_l)
print("Interval upper:", interval_u)

# Logic of the code
while error >= i_error:
    print("Iteration:", i)
    # Here we calcule the value of the functions
    f_a = function(interval_l)
    f_b = function(interval_u)
    print("Midpoint:",midpoint(interval_u, interval_l))
    f_m = function(midpoint(interval_u, interval_l))
    print("f(a):", f_a)
    print("f(b):", f_b)
    print("f(m):", f_m)
    error = (interval_u - interval_l)/2
    # New interval
    if (have_opposite_signs(f_a, f_m)):
        interval_u = midpoint(interval_u, interval_l)
    else:
        interval_l = midpoint(interval_u, interval_l)
    print("Interval lower:", interval_l)
    print("Interval upper:", interval_u)
    print("Error:", error)
    i += 1