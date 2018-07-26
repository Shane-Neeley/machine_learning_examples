import pandas as pd

l = pd.Series(["EGFR", "ALK", "BRAF", "EGFR3", "ALK"])

p = pd.get_dummies(l)


print(p)
