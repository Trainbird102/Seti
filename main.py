import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import math

reliability=0.9
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

def find_min_m_Version2(p, e):
    if p <= 0 or p >= 1:
        raise ValueError("p должно быть в интервале (0, 1)")
    if e <= 0:
        raise ValueError("e должно быть положительным числом")

    m = 1
    prev_product = 1 - p  # Произведение для m=1: (1 - p^1)

    while True:
        next_term = 1 - p ** (m + 1)
        next_product = prev_product * next_term
        difference = prev_product - next_product

        if difference < e:
            return m

        prev_product = next_product
        m += 1
# Пример использования
p = 0.1
e = 0.01
result = find_min_m_Version2(p, e)
print(f"Минимальное m: {result}")


M = find_min_m(reliability, epsilon)

# Примеры использования поиска минимального количества резервных путей
print(f"M =", M,f"Количество резервных путей для матрицы с надежностью канала P={reliability}")

#формула для вычисления заполнения матрицы горячего резервирования
def calculate_probability(reliability, m):
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
    if not (0 < reliability < 1):
        raise ValueError("p должно быть в диапазоне (0, 1)")
    if m < 1:
        raise ValueError("m должно быть >= 1")

    product = 1.0
    for k in range(1, m + 1):
        product *= (1 - reliability ** k)

    return 1 - product

#формула для вычисления заполнения матрицы холодного резервирования
def calculate_probability_logical(p, m):
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
    if not (0 < p < 1):
        raise ValueError("p должно быть в диапазоне (0, 1)")
    if m < 1:
        raise ValueError("m должно быть >= 1")

    total = 0.0
    for k in range(1, m + 1):
        p_k = p ** k
        term_inside_log = 1 - p_k
        if term_inside_log <= 0:
            raise ValueError(f"Аргумент логарифма <= 0 при k={k} (p={p})")
        log_value = math.log(term_inside_log)
        if log_value == 0:
            raise ValueError(f"Деление на ноль при k={k} (ln(1 - p^{k}) = 0)")
        total += 1 / log_value

    return -total

# Примеры использования:
print(calculate_probability(reliability, m=1))
print(calculate_probability(reliability, m=2))
print(calculate_probability(reliability, m=3))

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
G = create_reliable_graph(reliability=reliability)
#отображение графа
# plt.figure(figsize=(8, 6))
# nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
# plt.title(f'Связанный граф с 20 узлами и надежностью канала связи {reliability}')
# plt.show()

# Матрица смежности
adj_matrix = nx.adjacency_matrix(G).todense()
print("Матрица смежности графа:")
print(adj_matrix)

# Матрица кратчайших путей с ограничением по длине пути M
path_length = dict(nx.all_pairs_shortest_path_length(G))
n = len(G.nodes)
dist_matrix = np.zeros((n, n), dtype=int)

# Вычисляем M с помощью функции find_min_m
M = find_min_m(reliability, epsilon)
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
    plt.figure(figsize=(5, 5))

    # Собираем данные, исключая диагональ, нули и NaN
    x_data = []
    y_data = []

    for i in range(matrix_physical.shape[0]):
        for j in range(matrix_physical.shape[1]):
            if i != j:  # Исключаем диагональ
                phys_val = matrix_physical[i, j]
                log_val = matrix_logical[i, j]

                # Фильтруем нули и NaN
                if phys_val > 0 and not np.isnan(log_val) and log_val > 0:
                    x_data.append(phys_val)
                    y_data.append(log_val)


    # Построение графика
    plt.scatter(x_data, y_data, alpha=0.6, color='green', edgecolors='black')

    # Линия y=x для сравнения
    max_val = max(max(x_data), max(y_data)) * 1.1
    #plt.plot([0, max_val], [0, max_val], 'r--', label='y = x') окно с подписью на рисунке

    plt.title("Сравнение горячего и холодного резервирования")
    plt.xlabel("Физическая топология")
    plt.ylabel("Логическая топология")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle='--', alpha=0.5)
    #plt.legend()
    plt.show()

plot_physical_vs_logical(new_matrix_phisical, new_matrix_logical)

def plot_physical_vs_logical(matrix_physical,matrix_logical,error_percent=5,num_error_points=30,y_offset_percent=3):
    """
    Строит точечную диаграмму с заданным количеством точек в пределах погрешности.

    Параметры:
    matrix_physical (np.ndarray): Матрица физического резервирования
    matrix_logical (np.ndarray): Матрица логического резервирования
    error_percent (float): Процент погрешности по оси X
    num_error_points (int): Общее количество дополнительных точек
    y_offset_percent (float): Процент смещения по Y
    """
    plt.figure(figsize=(12, 12))

    # Сбор основных данных
    x_data, y_data = [], []
    for i in range(matrix_physical.shape[0]):
        for j in range(matrix_physical.shape[1]):
            if i != j and matrix_physical[i, j] > 0 and not np.isnan(matrix_logical[i, j]):
                x_data.append(matrix_physical[i, j])
                y_data.append(matrix_logical[i, j])

    if not x_data:
        raise ValueError("Нет данных для построения графика")

    # Расчет погрешностей
    x_errors = [x * error_percent / 100 for x in x_data]
    y_offsets = [y * y_offset_percent / 100 for y in y_data]

    # Генерация дополнительных точек (30 штук)
    np.random.seed(42)  # Для воспроизводимости
    x_scatter, y_scatter = [], []
    n_main = len(x_data)
    points_per_main = max(1, num_error_points // n_main)  # Точек на основную

    for x, y, x_err, y_off in zip(x_data, y_data, x_errors, y_offsets):
        # Случайные смещения в пределах погрешности
        x_rand = x + x_err * np.random.uniform(-1, 0, points_per_main)
        y_rand = y + y_off * np.random.uniform(-1, 1, points_per_main)
        x_scatter.extend(x_rand)
        y_scatter.extend(y_rand)

    # Обрезаем до 30 точек, если сгенерировано больше
    x_scatter = x_scatter[:num_error_points]
    y_scatter = y_scatter[:num_error_points]

    # Основные точки
    plt.scatter(
        x_data, y_data,
        s=80,
        color='navy',
        alpha=0.8,
        label='Основные данные',
        zorder=3
    )

    # Точки погрешности
    plt.scatter(
        x_scatter, y_scatter,
        s=40,
        color='orange',
        alpha=0.5,
        marker='x',
        label=f'Погрешность ({num_error_points} точек)',
        zorder=2
    )

    # Линия y=x
    max_val = max(max(x_data), max(y_data)) * 1.1
    plt.plot([0, max_val], [0, max_val], 'r--', label='y = x')

    plt.title("Сравнение горячего и холодного резервирования\n", fontsize=14)
    plt.xlabel("Физическая топология (горячий резерв)", fontsize=12)
    plt.ylabel("Логическая топология (холодный резерв)", fontsize=12)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.show()

plot_physical_vs_logical(new_matrix_phisical,new_matrix_logical,error_percent=15,num_error_points=30)
