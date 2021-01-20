# branching factor estimate is
# \[b = 8 G + 13 {G \choose 2} + 6  {G \choose 3} = G^3 + \frac{7}{2}G^2 + \frac{7}{2}G\]
# equtation in latex format
import operator as op
from functools import reduce

def ncr(n, r):
    if r > n:
        return 0
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom
def tool_branches(tool_name, g):
    if tool_name in ['Line', 'Perpendicular_Bisector', 'intersection']:
        return 0.5 * (g + g*g)
    if tool_name in ['Circle', 'Perpendicular', 'Parallel']:
        return g*g
    if tool_name in ['Compass', 'Angle_Bisector']:
        return 0.5*g*g*g + 0.5*g*g

def get_branching_factor(g, tools):
    b = 0
    for t in tools:
        b += int(tool_branches(t,g))
    return b

for i in range(10):
    b = get_branching_factor(i, ['Line', 'Perpendicular_Bisector', 'intersection','Circle', 'Perpendicular', 'Parallel','Compass', 'Angle_Bisector'])
    print(b)


