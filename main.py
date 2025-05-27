import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
import pandas as pd
import math


def find_min_m(reliability, epsilon):
    if not (0 < reliability < 1):
        raise ValueError("p должно быть в диапазоне (0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon должно быть > 0")

    m = 1
    prev_product = 1 - reliability ** 1

    while True:
        next_term = 1 - reliability ** (m + 1)
        next_product = prev_product * next_term
        numerator = prev_product * reliability ** (m + 1)
        denominator = 1 - prev_product

        if denominator == 0:
            return "Условие не выполняется (деление на ноль)"
        ratio = numerator / denominator

        if ratio < epsilon:
            return m
        else:
            prev_product = next_product
            m += 1


def calculate_probability(reliability, M):
    if not (0 < reliability < 1):
        raise ValueError("p должно быть в диапазоне (0, 1)")
    if M < 1:
        raise ValueError("m должно быть >= 1")

    product = 1.0
    for k in range(1, M + 1):
        product *= (1 - reliability ** k)
    return 1 - product


def calculate_probability_logical(reliability, m):
    if not (0 < reliability < 1):
        raise ValueError("p должно быть в диапазоне (0, 1)")
    if m < 1:
        raise ValueError("m должно быть >= 1")

    total = 0.0
    for k in range(1, m + 1):
        p_k = reliability ** k
        term_inside_log = 1 - p_k
        if term_inside_log <= 0:
            raise ValueError(f"Аргумент логарифма <= 0 при k={k}")
        log_value = math.log(term_inside_log)
        if log_value == 0:
            raise ValueError(f"Деление на ноль при k={k}")
        total += 1 / log_value
    return -total


def create_reliable_graph(num_nodes=15, reliability=0.1, additional_edges=2):
    """
    Создает связный граф с заданным количеством узлов и вероятностью соединения.

    Параметры:
        num_nodes (int): Количество вершин в графе
        reliability (float): Вероятность добавления ребра [0.0, 1.0]
        additional_edges (int): Целевое количество дополнительных ребер

    Возвращает:
        nx.Graph: Связный граф NetworkX
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # Гарантированно создаем связный граф (линейная цепочка)
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)

    # Добавляем дополнительные ребра с учетом вероятности
    edges_added = 0
    attempts = 0
    max_attempts = additional_edges * 5  # Защита от бесконечного цикла

    while edges_added < additional_edges and attempts < max_attempts:
        # Генерируем случайную пару узлов
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)

        if u != v and not G.has_edge(u, v):
            # Проверяем вероятность соединения
            if random.random() <= reliability:
                G.add_edge(u, v)
                edges_added += 1
            attempts += 1

    return G


class ReliabilityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ надёжности сетей")
        self.root.geometry("1200x800")

        self.reliability = tk.DoubleVar(value=0.5)
        self.epsilon = tk.DoubleVar(value=0.001)
        self.excel_var = tk.BooleanVar(value=True)

        self.create_widgets()
        self.setup_plots()

    def create_widgets(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(control_frame, text="Надёжность (p):").grid(row=0, column=0, padx=5)
        ttk.Entry(control_frame, textvariable=self.reliability, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Эпсилон (ε):").grid(row=0, column=2, padx=5)
        ttk.Entry(control_frame, textvariable=self.epsilon, width=10).grid(row=0, column=3, padx=5)

        ttk.Checkbutton(control_frame, text="Сохранять в Excel", variable=self.excel_var).grid(row=0, column=4, padx=10)
        ttk.Button(control_frame, text="Рассчитать", command=self.calculate).grid(row=0, column=5, padx=10)
        ttk.Button(control_frame, text="Выход", command=self.root.quit).grid(row=0, column=6, padx=10)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_graph_tab()
        self.create_matrix_tab()
        self.create_compare_tab()

    def create_graph_tab(self):
        self.graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_frame, text="Топология сети")
        self.canvas_graph = None

    def create_matrix_tab(self):
        self.matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.matrix_frame, text="Матрицы")

        self.matrix_text = scrolledtext.ScrolledText(
            self.matrix_frame,
            wrap=tk.NONE,
            font=('Courier New', 9),
            tabs=('0.5i', '1i', '2i')
        )
        hscroll = ttk.Scrollbar(self.matrix_frame, orient=tk.HORIZONTAL, command=self.matrix_text.xview)
        self.matrix_text.configure(xscrollcommand=hscroll.set)
        hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.matrix_text.pack(fill=tk.BOTH, expand=True)

    def create_compare_tab(self):
        self.compare_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.compare_frame, text="Сравнение")
        self.canvas_compare = None

    def setup_plots(self):
        plt.style.use('ggplot')

    def calculate(self):
        try:
            p = self.reliability.get()
            eps = self.epsilon.get()

            if not (0 < p < 1) or eps <= 0:
                raise ValueError("Некорректные входные параметры")

            M = find_min_m(p, eps)
            G = create_reliable_graph(reliability=p)

            self.update_graph(G)
            matrices = self.calculate_matrices(G, p, M)
            self.update_matrices(*matrices)
            self.update_comparison_plot(*matrices[2:])

            if self.excel_var.get():
                self.save_to_excel(*matrices)

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.clear_all()

    def calculate_matrices(self, G, p, M):
        n = len(G.nodes)
        adj_matrix = nx.adjacency_matrix(G).todense()

        path_length = dict(nx.all_pairs_shortest_path_length(G))
        dist_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if j in path_length[i] and path_length[i][j] <= M:
                    dist_matrix[i, j] = path_length[i][j]

        phys_matrix = np.zeros((n, n))
        log_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] > 0:
                    phys_matrix[i, j] = calculate_probability(p, dist_matrix[i, j])
                    try:
                        log_matrix[i, j] = calculate_probability_logical(p, dist_matrix[i, j])
                    except:
                        log_matrix[i, j] = np.nan

        return adj_matrix, dist_matrix, phys_matrix, log_matrix

    def update_graph(self, G):
        if self.canvas_graph:
            self.canvas_graph.get_tk_widget().destroy()

        fig = plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True,
                node_color='lightblue',
                edge_color='gray',
                node_size=500,
                font_size=8)
        plt.title(f"Топология сети (p={self.reliability.get():.2f}, узлов={len(G.nodes)})")

        self.canvas_graph = FigureCanvasTkAgg(fig, self.graph_frame)
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_matrices(self, adj, dist, phys, log):
        nodes = list(range(len(adj)))

        def format_matrix(matrix, precision=3, zero_replace="0"):
            df = pd.DataFrame(matrix, index=nodes, columns=nodes)
            return df.to_string(float_format=lambda x: f"{x:.{precision}f}" if x != 0 else zero_replace,
                                line_width=300, max_rows=20, max_cols=20)

        text = "=" * 50 + " МАТРИЦА СМЕЖНОСТИ " + "=" * 50 + "\n\n"
        text += format_matrix(adj, 0, " ") + "\n\n"

        text += "=" * 50 + " МАТРИЦА КРАТЧАЙШИХ ПУТЕЙ " + "=" * 50 + "\n\n"
        text += format_matrix(dist, 0, " ") + "\n\n"

        text += "=" * 50 + " ФИЗИЧЕСКОЕ РЕЗЕРВИРОВАНИЕ " + "=" * 50 + "\n\n"
        text += format_matrix(phys) + "\n\n"

        text += "=" * 50 + " ЛОГИЧЕСКОЕ РЕЗЕРВИРОВАНИЕ " + "=" * 50 + "\n\n"
        text += format_matrix(log) + "\n"

        self.matrix_text.delete(1.0, tk.END)
        self.matrix_text.insert(tk.END, text)

    def update_comparison_plot(self, phys, log):
        if self.canvas_compare:
            self.canvas_compare.get_tk_widget().destroy()

        fig = plt.figure(figsize=(8, 6))

        # Собираем данные для графика
        x = phys.flatten()
        y = log.flatten()
        valid_indices = ~np.isnan(y)
        x = x[valid_indices]
        y = y[valid_indices]

        plt.scatter(x, y, alpha=0.5, color='green', label='Точки данных')
        plt.plot([0, 1], [0, 1], 'r--', label='Линия равенства')

        # Добавляем координаты для 10 случайных точек
        indices = np.random.choice(len(x), size=min(10, len(x)), replace=False)
        for i in indices:
            plt.annotate(f'({x[i]:.2f}, {y[i]:.2f})',
                         (x[i], y[i]),
                         textcoords="offset points",
                         xytext=(5, 5),
                         ha='left',
                         fontsize=8,
                         arrowprops=dict(arrowstyle='->', color='gray'))

        plt.title("Сравнение методов резервирования")
        plt.xlabel("Физическое резервирование")
        plt.ylabel("Логическое резервирование")
        plt.grid(True)
        plt.legend()

        self.canvas_compare = FigureCanvasTkAgg(fig, self.compare_frame)
        self.canvas_compare.draw()
        self.canvas_compare.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_to_excel(self, adj, dist, phys, log):
        try:
            excel_filename = "network_analysis.xlsx"
            with pd.ExcelWriter(excel_filename) as writer:
                pd.DataFrame(adj).to_excel(writer, sheet_name='Смежность')
                pd.DataFrame(dist).to_excel(writer, sheet_name='Пути')
                pd.DataFrame(phys).to_excel(writer, sheet_name='Физическое')
                pd.DataFrame(log).to_excel(writer, sheet_name='Логическое')
            messagebox.showinfo("Сохранено", f"Данные сохранены в {excel_filename}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def clear_all(self):
        if self.canvas_graph:
            self.canvas_graph.get_tk_widget().destroy()
        if self.canvas_compare:
            self.canvas_compare.get_tk_widget().destroy()
        self.matrix_text.delete(1.0, tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = ReliabilityApp(root)
    root.mainloop()