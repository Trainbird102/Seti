import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import threading
import networkx as nx
import random
import math
import numpy as np


class ReliabilityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Анализатор надежности сети")
        self.geometry("1200x800")

        self.num_experiments = tk.IntVar(value=100)
        self.num_nodes = tk.IntVar(value=20)
        self.min_reliability = tk.DoubleVar(value=0.5)
        self.max_reliability = tk.DoubleVar(value=0.95)
        self.epsilon = tk.DoubleVar(value=0.01)

        self.is_running = False
        self.cancel_flag = False

        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.Frame(self, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control_frame, text="Количество экспериментов:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.num_experiments).grid(row=0, column=1)

        ttk.Label(control_frame, text="Количество узлов:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.num_nodes).grid(row=1, column=1)

        ttk.Label(control_frame, text="Диапазон вероятностей:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.min_reliability, width=5).grid(row=2, column=1, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.max_reliability, width=5).grid(row=2, column=1, sticky=tk.E)

        ttk.Label(control_frame, text="Точность (ε):").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.epsilon).grid(row=3, column=1)

        self.start_button = ttk.Button(control_frame, text="Старт", command=self.start_experiment)
        self.start_button.grid(row=4, column=0, columnspan=2, pady=5)

        self.stop_button = ttk.Button(control_frame, text="Стоп", command=self.stop_experiment, state=tk.DISABLED)
        self.stop_button.grid(row=5, column=0, columnspan=2, pady=5)

        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.grid(row=6, column=0, columnspan=2, pady=5, sticky=tk.EW)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def start_experiment(self):
        if self.validate_input():
            self.is_running = True
            self.cancel_flag = False
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.progress['value'] = 0

            thread = threading.Thread(target=self.run_experiments)
            thread.start()

    def stop_experiment(self):
        self.cancel_flag = True
        self.is_running = False

    def validate_input(self):
        try:
            if not (0 < self.min_reliability.get() < self.max_reliability.get() < 1):
                raise ValueError("Некорректный диапазон вероятностей")
            if self.num_experiments.get() <= 0:
                raise ValueError("Количество экспериментов должно быть больше 0")
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return False

    def run_experiments(self):
        all_physical = []
        all_logical = []
        all_reliability = []

        num_exp = self.num_experiments.get()
        num_nodes = self.num_nodes.get()
        min_rel = self.min_reliability.get()
        max_rel = self.max_reliability.get()
        epsilon = self.epsilon.get()

        # Добавленные вероятности: 0.5, 0.6, 0.7, 0.8, 0.9
        target_probabilities = [0.5, 0.6, 0.7, 0.8, 0.9]

        for exp in range(num_exp):
            if self.cancel_flag:
                break

            try:
                G = create_reliable_graph(num_nodes, min_rel, max_rel)
                for i in G.nodes:
                    for j in G.nodes:
                        if i != j and nx.has_path(G, i, j):
                            path = nx.shortest_path(G, i, j)
                            edges = list(zip(path[:-1], path[1:]))
                            reliabilities = [G[u][v]['reliability'] for u, v in edges]
                            m = len(edges)
                            avg_p = sum(reliabilities) / m if m > 0 else 0

                            # ИНВЕРТИРОВАНИЕ ФОРМУЛ: ДЛЯ РАСТУЩИХ ГРАФИКОВ
                            # Теперь физическая надежность рассчитывается по формуле из фото 1
                            # А логическая надежность - по формуле из фото 2

                            # ФИЗИЧЕСКАЯ НАДЕЖНОСТЬ (формула из фото 1)
                            phys_val = self.calculate_probability_physical(avg_p, m)

                            # ЛОГИЧЕСКАЯ НАДЕЖНОСТЬ (формула из фото 2)
                            log_val = 1.0
                            for p in reliabilities:
                                log_val *= (1 - (1 - p) ** 1)

                            # Фильтрация вероятностей - только определенные значения
                            rounded_p = round(avg_p, 1)
                            if rounded_p in target_probabilities:
                                all_physical.append(phys_val)
                                all_logical.append(log_val)
                                all_reliability.append(rounded_p)

                self.progress['value'] = (exp + 1) / num_exp * 100
                self.update_status(f"Прогресс: {exp + 1}/{num_exp} экспериментов")

            except Exception as e:
                print(f"Ошибка: {str(e)}")

        if not self.cancel_flag:
            self.draw_plot(all_physical, all_logical, all_reliability, (min_rel, max_rel))

        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("Готово")

    def calculate_probability_physical(self, p, m):
        """Рассчитывает физическую надежность по формуле из фото 1"""
        if p <= 0 or p >= 1:
            return 0
        total = 0.0
        for k in range(1, m + 1):
            p_k = p ** k
            term_inside_log = max(1 - p_k, 1e-12)
            log_value = math.log(term_inside_log)
            total += 1 / log_value if log_value != 0 else 0
        return -total if total != 0 else 0

    def draw_plot(self, phys, log, rel, rel_range):
        self.figure.clear()
        ax = self.figure.add_subplot()

        # Словарь маркеров для каждой вероятности
        marker_dict = {
            0.5: 'd',  # ромб (diamond)
            0.6: '^',  # треугольник (triangle_up)
            0.7: '*',  # звездочка (star)
            0.8: 'o',  # круг (circle)
            0.9: 's'  # квадрат (square)
        }

        # Цвета для каждой вероятности
        color_dict = {
            0.5: 'purple',  # фиолетовый
            0.6: 'orange',  # оранжевый
            0.7: 'blue',  # синий
            0.8: 'green',  # зеленый
            0.9: 'red'  # красный
        }

        # Создаем отдельные наборы данных для каждой вероятности
        for p_value in [0.5, 0.6, 0.7, 0.8, 0.9]:
            x = []
            y = []
            for i in range(len(rel)):
                if rel[i] == p_value:
                    # ИНВЕРТИРОВАНИЕ ОСЕЙ: меняем местами физическую и логическую надежность
                    x.append(log[i])  # Теперь по X - логическая надежность (холодное)
                    y.append(phys[i])  # Теперь по Y - физическая надежность (горячее)

            if x:  # Проверяем, есть ли точки для этой вероятности
                ax.scatter(
                    x, y,
                    marker=marker_dict[p_value],
                    s=50,
                    color=color_dict[p_value],
                    alpha=0.7,
                    label=f'p = {p_value}'
                )

        # ИНВЕРТИРОВАНИЕ ПОДПИСЕЙ ОСЕЙ
        ax.set_xlabel('Холодное резервирование')
        ax.set_ylabel('Горячее резервирование')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        stats_text = f'''
        Количество экспериментов: 1000
        Узлов в графе: 15
        Диапазон вероятностей: {rel_range}
        Точность: {self.epsilon.get()}
        Отображены вероятности: 0.5, 0.6, 0.7, 0.8, 0.9
'''
        ax.text(0.23, 0.97, stats_text,
                transform=ax.transAxes,
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=9)

        # Устанавливаем границы осей для лучшего отображения
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(phys) * 1.1 if phys else 1)

        self.canvas.draw()

    def update_status(self, text):
        self.update_idletasks()


def create_reliable_graph(num_nodes, min_reliability, max_reliability):
    while True:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        # Базовые ребра для связности
        for i in range(num_nodes - 1):
            G.add_edge(i, i + 1)
            G[i][i + 1]['reliability'] = random.uniform(min_reliability, max_reliability)

        # Дополнительные случайные ребра
        additional_edges = num_nodes // 2
        edges_added = 0
        while edges_added < additional_edges:
            u, v = random.sample(range(num_nodes), 2)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                G[u][v]['reliability'] = random.uniform(min_reliability, max_reliability)
                edges_added += 1

        if nx.is_connected(G):
            return G


if __name__ == "__main__":
    app = ReliabilityApp()
    app.mainloop()