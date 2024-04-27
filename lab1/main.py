import numpy as np
import matplotlib.pyplot as plt
import locale
from scipy.stats import poisson

# locale.setlocale(locale.LC_NUMERIC, "de_DE")
# plt.rcdefaults()
# plt.rcParams["axes.formatter.use_locale"] = True

# генерация случайной величины согласно распредлению Пуассона
values = poisson.rvs(mu=197.3, size=9000)  # mu это лямбда
np.savetxt("data.csv", values, delimiter=",") # cохранение
array = values.tolist()

plt.hist(
    values,
    bins=np.arange(min(values), max(values)),
    density=True,
)
plt.title("Распределение Пуассона")
plt.xlabel("Значение величины")
plt.ylabel("Частота")

# Плотность вероятности
x = np.arange(0, max(values))
pmf = poisson.pmf(x, mu=197.3)
plt.plot(x, pmf, "r-", linewidth=2, label="Плотность вероятности")
plt.show()

print(f'Среднее арифметическое: {np.average(array):.3f}\nМедиана: {np.median(array):.3f}\nДисперсия: {np.var(array):.3f}')

