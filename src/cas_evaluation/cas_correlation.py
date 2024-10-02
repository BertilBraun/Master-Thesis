import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def parse_log_file(log_file_path):
    pattern = r'Profile score: (\d+\.\d+) With given: (\d+) abstracts and (\d+) jahresabschlüsse'
    data = []

    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                score = float(match.group(1))
                abstracts = int(match.group(2))
                jahresabschluesse = int(match.group(3))
                # if abstracts + jahresabschluesse > 30:
                #     continue
                data.append([score, abstracts, jahresabschluesse])

    return pd.DataFrame(data, columns=['Score', 'Abstracts', 'Jahresabschlüsse'])


def calculate_correlations(df):
    correlation_abs_score = pearsonr(df['Abstracts'], df['Score'])[0]
    correlation_jahr_score = pearsonr(df['Jahresabschlüsse'], df['Score'])[0]
    correlation_combined_score = pearsonr(df['Abstracts'] + df['Jahresabschlüsse'], df['Score'])[0]

    return correlation_abs_score, correlation_jahr_score, correlation_combined_score


def plot_correlations(df):
    # Add a new column for combined abstracts and jahresabschlüsse
    df['Combined'] = df['Abstracts'] + df['Jahresabschlüsse']

    # Scatter plots
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.scatterplot(x='Abstracts', y='Score', data=df)
    plt.title('Correlation between Abstracts and Score')

    plt.subplot(1, 3, 2)
    sns.scatterplot(x='Jahresabschlüsse', y='Score', data=df)
    plt.title('Correlation between Jahresabschlüsse and Score')

    plt.subplot(1, 3, 3)
    sns.scatterplot(x='Combined', y='Score', data=df)
    plt.title('Correlation between Abstracts + Jahresabschlüsse and Score')

    plt.tight_layout()
    plt.show()

    # Heatmap of correlations
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()


if __name__ == '__main__':
    PATH1 = R'C:\Users\berti\OneDrive\Docs\Studium\Semester 8\Masterarbeit\Master-Thesis\logs\2024-08-27'
    PATH2 = R'C:\Users\berti\OneDrive\Docs\Studium\Semester 8\Masterarbeit\Master-Thesis\logs\2024-08-28'

    accumulated_df = pd.DataFrame(columns=['Score', 'Abstracts', 'Jahresabschlüsse'])

    for log_file_path in [os.path.join(PATH1, log_file) for log_file in os.listdir(PATH1)] + [
        os.path.join(PATH2, log_file) for log_file in os.listdir(PATH2)
    ]:
        if not os.path.isfile(log_file_path) or not log_file_path.endswith('.log'):
            continue

        df = parse_log_file(log_file_path)

        if len(df) <= 10:
            continue

        corr_abs_score, corr_jahr_score, corr_combined_score = calculate_correlations(df)

        print(f'File: {log_file_path}')

        print(f'Korrelation zwischen Abstracts und Score: {corr_abs_score}')
        print(f'Korrelation zwischen Jahresabschlüsse und Score: {corr_jahr_score}')
        print(f'Korrelation zwischen Abstracts + Jahresabschlüsse und Score: {corr_combined_score}')

        print()
        print(f'Mean Number of Abstracts: {df["Abstracts"].mean()}')
        print(f'Mean Number of Jahresabschlüsse: {df["Jahresabschlüsse"].mean()}')
        print()

        print(f'Mean Score: {df["Score"].mean()}')
        print(f'Median Score: {df["Score"].median()}')
        print(f'Min Score: {df["Score"].min()}')
        print(f'Max Score: {df["Score"].max()}')
        print(f'Std Score: {df["Score"].std()}')
        print(f'Number of Scores: {len(df)}')
        print('-' * 50)

        accumulated_df = pd.concat([accumulated_df, df], ignore_index=True)

        plot_correlations(df)
    plot_correlations(accumulated_df)
