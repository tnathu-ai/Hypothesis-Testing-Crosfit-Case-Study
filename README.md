### A comprehensive guide to perform hypothesis tests and find confidence interval, degrees of freedom, critical value, test statistic

<p align="center">
<a href="https://tnathu-ai.medium.com/parametric-and-non-parametric-tests-case-study-in-python-1b647c1df3af" target="blank">Read my Medium<img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/medium.svg" alt="@tnathu-ai" height="30" width="40" /></a>
</p>

![tree map for summary](media/images/hypothesis-testing.png)

# Open Discussion
+ Non-parametric hypothesis test has the assumption of a continuous and symmetric distribution's mean(median)

Regardless of The central limit theorem states that the distribution of sample means equal to or greater than 30 are often considered sufficient for the normal distribution. What if my distribution does not assume I am sampling from a particular distribution and is either continuous or symmetric about the mean (e.g. skewed distribution)? What type of hypothesis testing should I use?

# Folder Structure & Architecture

```
├── LICENSE
├── README.md
├── data
│   ├── external
│   │   ├── athletes.csv
│   │   ├── leaderboard_15.csv
│   │   └── source.txt
│   ├── interim
│   │   ├── cleaned_data.csv
│   │   └── source.txt
│   └── processed
├── media
│   ├── images
│   │   ├── CI_2_Proportions.png
│   │   ├── CI_variance.png
│   │   ├── CLT_1_2.png
│   │   ├── CLT_2.png
│   │   ├── CLT_definition.png
│   │   ├── The_Central_Limit_Theorem.png
│   │   ├── chi_squared_distribution.png
│   │   ├── hypothesis-testing.png
│   │   ├── hypothesis_test_2_proportions.png
│   │   ├── test_statistic_single_variance.png
│   │   └── tests_for_Independence.png
│   └── plots
│       ├── box_plot_score_by_regions.png
│       ├── boxplot_distribution_scores_by_regions.png
│       ├── distplot.png
│       ├── female_male_weight_hist.png
│       ├── heatmap to indicates correlation between variables.png
│       ├── hist.png
│       ├── missing_plot.png
│       ├── sample_distplot.png
│       └── visualize_word.png
├── notebooks
│   ├── EDA.ipynb
│   ├── EDA.py
│   ├── demo_normality_test.ipynb
│   ├── demo_normality_test.py
│   ├── model.ipynb
│   ├── model.py
│   ├── model_drop_cols.ipynb
│   ├── model_drop_cols.py
│   ├── normality_test.ipynb
│   ├── normality_test.py
│   ├── pop_distplot.png
│   ├── preprocessing.ipynb
│   ├── preprocessing.py
│   ├── quiz_2.ipynb
│   ├── quiz_2.py
│   ├── quiz_3.ipynb
│   ├── quiz_3.py
│   ├── regression.ipynb
│   ├── regression.py
│   ├── statistical_methods.ipynb
│   └── statistical_methods.py
└── src
```
