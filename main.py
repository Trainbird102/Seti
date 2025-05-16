import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import math

reliability=0.5
epsilon = 0.01

#поиск минимального значение количества резервных путей
def find_min_m(reliability, epsilon):
    """
        Находит минимальное значение m, удовлетворяющее условию:
        |prod(1-p^k)_{k=1}^m - prod(1-p^k)_{k=1}^{m+1}| / (1 - prod(1-p^k)_{k=1}^m) < epsilon

        Параметры:
        p (float): Вероятность, 0 < p < 1
        epsilon (float): Порог, epsilon > 0

        Возвращает:
        int: Минимальное m
        """
    if not (0 < reliability < 1):
        raise ValueError("p должно быть в диапазоне (0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon должно быть > 0")

    m = 1
    prev_product = 1 - reliability ** 1  # Начальное значение для m=1

    while True:
        # Вычисляем произведение для m+1
        next_term = 1 - reliability ** (m + 1)
        next_product = prev_product * next_term

        # Вычисляем числитель и знаменатель
        numerator = prev_product * reliability ** (m + 1)
        denominator = 1 - prev_product

        # Проверяем условие
        if denominator == 0:
            return "Условие не выполняется (деление на ноль)"

        ratio = numerator / denominator

        if ratio < epsilon:
            return m
        else:
            prev_product = next_product
            m += 1

def find_min_m_Version2(reliability, epsilon):
    if reliability <= 0 or reliability >= 1:
        raise ValueError("p должно быть в интервале (0, 1)")
    if epsilon <= 0:
        raise ValueError("e должно быть положительным числом")

    m = 1
    prev_product = 1 - reliability  # Произведение для m=1: (1 - p^1)

    while True:
        next_term = 1 - reliability ** (m + 1)
        next_product = prev_product * next_term
        difference = prev_product - next_product

        if difference < epsilon:
            return m

        prev_product = next_product
        m += 1

result = find_min_m_Version2(reliability, epsilon)
print(f"Минимальное m: {result}")


M = find_min_m_Version2(reliability, epsilon)

# Примеры использования поиска минимального количества резервных путей
print(f"M =", M,f"Количество резервных путей для матрицы с надежностью канала P={reliability}")

#формула для вычисления заполнения матрицы горячего резервирования
reliability1 = 0.5
def calculate_probability(reliability1, M):
    """
    Вычисляет вероятность P по формуле:
    P = 1 - ∏_{k=1}^{m} (1 - p^k)

    Параметры:
    p (float): Вероятность, 0 < p < 1
    m (int): Целое число, m >= 1

    Возвращает:
    float: Значение P

    Исключения:
    ValueError: Если p не в диапазоне (0, 1) или m < 1
    """
    if not (0 < reliability1 < 1):
        raise ValueError("p должно быть в диапазоне (0, 1)")
    if M < 1:
        raise ValueError("m должно быть >= 1")

    product = 1.0
    for k in range(1, M + 1):
        product *= (1 - reliability1 ** k)

    return 1 - product

#формула для вычисления заполнения матрицы холодного резервирования
def calculate_probability_logical(reliability1, m):
    """
    Вычисляет значение формулы: -Σ_{k=1}^m [1 / ln(1 - p^k)]

    Параметры:
    p (float): Вероятность, 0 < p < 1
    m (int): Целое число, m >= 1

    Возвращает:
    float: Результат вычисления

    Исключения:
    ValueError: Если входные данные некорректны
    """
    if not (0 < reliability1 < 1):
        raise ValueError("p должно быть в диапазоне (0, 1)")
    if m < 1:
        raise ValueError("m должно быть >= 1")

    total = 0.0
    for k in range(1, m + 1):
        p_k = reliability1 ** k
        term_inside_log = 1 - p_k
        if term_inside_log <= 0:
            raise ValueError(f"Аргумент логарифма <= 0 при k={k} (p={reliability1})")
        log_value = math.log(term_inside_log)
        if log_value == 0:
            raise ValueError(f"Деление на ноль при k={k} (ln(1 - p^{k}) = 0)")
        total += 1 / log_value

    return -total

# Примеры использования:
print(calculate_probability(reliability, M=1))
print(calculate_probability(reliability, M=2))
print(calculate_probability(reliability, M=3))
print(calculate_probability(reliability, M=4))

# Функция для создания связанного графа с учетом надежности канала связи
def create_reliable_graph(num_nodes=20, base_edges=19, additional_edges=10, reliability=0.9):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # Создаем цепочку для связности с вероятностью reliability
    for i in range(base_edges):
        if random.random() <= reliability:
            G.add_edge(i, i + 1)

    # Добавляем дополнительные случайные рёбра с учетом надежности
    edges_added = 0
    while edges_added < additional_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and not G.has_edge(u, v) and random.random() <= reliability:
            G.add_edge(u, v)
            edges_added += 1

    return G

# Создаем граф с указанной надежностью канала связи
G = create_reliable_graph(reliability=reliability1)
# отображение графа
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title(f'Связанный граф с 20 узлами и надежностью канала связи {reliability}')
plt.show()

# Матрица смежности
adj_matrix = nx.adjacency_matrix(G).todense()
print("Матрица смежности графа:")
print(adj_matrix)

# Матрица кратчайших путей с ограничением по длине пути M
path_length = dict(nx.all_pairs_shortest_path_length(G))
n = len(G.nodes)
dist_matrix = np.zeros((n, n), dtype=int)

# Вычисляем M с помощью функции find_min_m
M = find_min_m_Version2(reliability, epsilon)
print(f"Значение M: {M}")

# Записываем только пути, не превышающие M
for i in range(n):
    for j in range(n):
        if j in path_length[i] and path_length[i][j] <= M:
            dist_matrix[i, j] = path_length[i][j]
        else:
            dist_matrix[i, j] = 0  # Записываем 0, если путь превышает M или отсутствует

print("Матрица длины кратчайших путей с ограничением по M:")
print(dist_matrix)

# Новая матрица по формуле горячего резервирования
new_matrix_phisical = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            new_matrix_phisical[i, j] = 0
        elif dist_matrix[i, j] > 0:
            new_matrix_phisical[i, j] = calculate_probability(reliability, dist_matrix[i, j])
        else:
            new_matrix_phisical[i, j] = 0.0
print("\nМатрица резервных путей физическая топология:")
print(new_matrix_phisical)

# Новая матрица по формуле холодного резервирования
new_matrix_logical = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            new_matrix_logical[i, j] = 0.0  # Для диагональных элементов
        elif dist_matrix[i, j] > 0:
            try:
                new_matrix_logical[i, j] = calculate_probability_logical(reliability, dist_matrix[i, j])
            except ValueError as e:
                print(f"Ошибка при вычислении для узлов {i}-{j}: {e}")
                new_matrix_logical[i, j] = float('nan')  # В случае ошибки ставим NaN
        else:
            new_matrix_logical[i, j] = 0.0  # Для узлов без пути
print("\nМатрица резервных путей логическая топология:")
print(new_matrix_logical)

# Сохраняем матрицы в Excel
excel_filename = "matrices_data.xlsx"
with pd.ExcelWriter(excel_filename) as writer:
    pd.DataFrame(adj_matrix).to_excel(writer, sheet_name='Матрица смежности')
    pd.DataFrame(dist_matrix).to_excel(writer, sheet_name='Матрица длин кратчайших путей ')
    pd.DataFrame(new_matrix_phisical).to_excel(writer, sheet_name='Матрица горячего резервирования (физическая)')
    pd.DataFrame(new_matrix_logical).to_excel(writer, sheet_name='Матрица холодного резервирования (логическая)')

print(f"Матрицы успешно сохранены в файл {excel_filename}")

# Визуализация матриц и графа

# plt.figure(figsize=(8, 7))
# plt.title("Матрица смежности")
# plt.imshow(adj_matrix, cmap='Blues')
# plt.colorbar(label='Связь')
# plt.xticks(np.arange(n), np.arange(n))
# plt.yticks(np.arange(n), np.arange(n))
# plt.xlabel("Узел")
# plt.ylabel("Узел")
# plt.show()

# plt.figure(figsize=(8, 7))
# plt.title("Матрица длин кратчайших путей")
# plt.imshow(dist_matrix, cmap='Oranges')
# plt.colorbar(label='Длина пути')
# plt.xticks(np.arange(n), np.arange(n))
# plt.yticks(np.arange(n), np.arange(n))
# plt.xlabel("Узел")
# plt.ylabel("Узел")
# plt.show()

# plt.figure(figsize=(8, 7))
# plt.title("Новая матрица с замененными значениями")
# plt.imshow(new_matrix_phisical, cmap='Greens')
# plt.colorbar(label='Значение')
# plt.xticks(np.arange(n), np.arange(n))
# plt.yticks(np.arange(n), np.arange(n))
# plt.xlabel("Узел")
# plt.ylabel("Узел")
# plt.show()

# plt.figure(figsize=(8, 6))
# nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
# plt.title(f'Связанный граф с 20 узлами и надежностью канала связи {reliability}')
# plt.show()

def plot_physical_vs_logical(matrix_physical, matrix_logical):
    """
    Строит точечную диаграмму для сравнения значений матриц резервирования.

    Параметры:
    matrix_physical (np.ndarray): Матрица физического резервирования (N x N)
    matrix_logical (np.ndarray): Матрица логического резервирования (N x N)
    """
    plt.figure(figsize=(8, 8))

    # Собираем данные, исключая диагональ, нули и NaN
    x_data = []
    y_data = []
    coordinates = []  # Для хранения координат

    for i in range(matrix_physical.shape[0]):
        for j in range(matrix_physical.shape[1]):
            if i != j:
                phys_val = matrix_physical[i, j]
                log_val = matrix_logical[i, j]

                if phys_val > 0 and not np.isnan(log_val) and log_val > 0:
                    x_data.append(phys_val)
                    y_data.append(log_val)
                    coordinates.append((phys_val, log_val))

    # Построение графика
    scatter = plt.scatter(x_data, y_data, alpha=0.6, color='green', edgecolors='black', label='Точки данных')

    # Добавление координат к точкам
    for x, y in coordinates:
        plt.text(x + 0.02,
                 y,
                 f'({x:.3f}, {y:.3f})',
                 fontsize=12,
                 alpha=0.7,
                 va='center',
                 rotation=0)

    # Линия y=x для сравнения
    max_val = max(max(x_data), max(y_data)) * 1.1 if x_data and y_data else 1
    plt.plot([0, max_val], [0, max_val], 'r--', label='y = x')

    plt.title("Сравнение горячего и холодного резервирования")
    plt.xlabel("Физическая топология")
    plt.ylabel("Логическая топология")
    plt.xlim(0, 1.25)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle='--', alpha=0.5)
    #plt.legend()
    plt.show()

plot_physical_vs_logical(new_matrix_phisical, new_matrix_logical)
