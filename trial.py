import pandas as pd

xlsx_file = "/content/Credit data.xlsx"  # Update the path if needed
df = pd.read_excel(xlsx_file)
print(df.head())  # Check if data is loaded correctly
