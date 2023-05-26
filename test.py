# classes = ["T", "A", "C", "H","a", "b", "1", "2", "-1", "-2"]
import pandas as pd

datalist = pd.read_csv(
    "valid_random_strings.csv",
)
print(datalist["Layouts"])
