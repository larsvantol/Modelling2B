import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

y = []
for _ in tqdm(range(int(1e6))):
    x = -1
    while x < 0:
        x = np.random.normal(0.1, 0.075)
    y.append(x)

diff = abs(np.mean(y) - 0.1)
y = [x - diff for x in y]


# Check if there are negative values
def neg_in_list(list):
    count = 0
    for x in list:
        if x < 0:
            count += 1
    return count


while neg_in_list(y) > 0:
    new_list = []
    print(neg_in_list(y))
    for x in tqdm(y):
        while x < 0:
            x = np.random.normal(0.1, 0.075)
        new_list.append(x)

    diff = abs(np.mean(new_list) - 0.1)
    y = [x - diff for x in new_list]

print(f"Mean: {np.mean(y)}")
print(f"Std: {np.std(y)}")

y_2 = []
for _ in tqdm(range(int(1e6))):
    x = -1
    while x < 0:
        x = abs(np.random.normal(0.1, 0.075))
    y_2.append(x)

plt.hist(y, bins=100)
plt.hist(y_2, bins=100, alpha=0.2)

plt.show()
