import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

values = np.loadtxt('data.csv', delimiter=',')

# Round the values and calculate min and max
values_rounded = np.round(values)
values_min = values_rounded.min().astype(int)
values_max = values_rounded.max().astype(int)

# Get the observed frequencies without normalizing to a probability density
observed_counts, bins = np.histogram(values_rounded, bins=range(values_min, values_max + 2))

# Calculate expected frequencies (as counts)
expected_counts = [stats.poisson.pmf(x, values.mean()) * len(values_rounded) for x in range(values_min, values_max + 1)]

# Perform chi-square test
chi_square_statistic, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)


# Calculate the total observed counts
total_observed = observed_counts.sum()

# Calculate expected frequencies (as counts) and adjust to match total observed counts
raw_expected_counts = [stats.poisson.pmf(x, values.mean()) * len(values_rounded) for x in range(values_min, values_max + 1)]
total_expected = sum(raw_expected_counts)
adjustment_factor = total_observed / total_expected
adjusted_expected_counts = [count * adjustment_factor for count in raw_expected_counts]

# Perform chi-square test with adjusted expected counts
chi_square_statistic, p_value = stats.chisquare(f_obs=observed_counts, f_exp=adjusted_expected_counts)

print("Chi-Square Statistic:", chi_square_statistic)
print("P-value:", p_value)

# Plot observed and expected frequencies
plt.figure(figsize=(10, 5))
plt.bar(bins[:-1], observed_counts, width=0.7, label='Observed frequencies', color='blue')
plt.plot(range(values_min, values_max + 1), expected_counts, 'o-', label='Expected frequencies', color='red')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Chi-Squared Test')
plt.show()
