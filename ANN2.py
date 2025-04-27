# Generate ANDNOT function using McCulloch-Pitts neural net by a python program.


def mcculloch_pitts_andnot(x1, x2):
    weights = [1, -1]
    threshold = 1
    net_input = (x1 * weights[0]) + (x2 * weights[1])
    
    return 1 if net_input >= threshold else 0


test_cases = [(0,0), (0,1), (1,0), (1,1)]
for x1, x2 in test_cases:
    print(f"ANDNOT({x1}, {x2}) = {mcculloch_pitts_andnot(x1, x2)}")