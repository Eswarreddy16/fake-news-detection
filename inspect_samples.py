import pandas as pd
import random

# Load both files
print("Sample of real.csv (first 3 rows):")
df_real = pd.read_csv('real.csv')
print(df_real['text'].head(3))
print("\nSample of Fake.csv (first 3 rows):")
df_fake = pd.read_csv('Fake.csv')
print(df_fake['text'].head(3))

# Print some random samples
print("\nRandom samples from real.csv:")
for _ in range(3):
    idx = random.randint(0, len(df_real)-1)
    print(f"\nReal article {idx}:")
    print(df_real.iloc[idx]['text'][:500], "...")

print("\nRandom samples from Fake.csv:")
for _ in range(3):
    idx = random.randint(0, len(df_fake)-1)
    print(f"\nFake article {idx}:")
    print(df_fake.iloc[idx]['text'][:500], "...")