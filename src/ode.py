import numpy as np


# -----------------------------------------------------------------------------
# Таблицы Бутчера для методов из пособия:
# 26 — midpoint
# 27 — Heun (explicit trapezoid)
# 28 — third-order Heun
# 29 — third-order Simpson-type RK
# 30 — classical RK4
# 31 — RK4 3/8-rule
# -----------------------------------------------------------------------------
"""
Для каждого метода заданы матрица A (коэффициенты стадий), векторы b (веса) и c (узлы), а также порядок точности.
"""
_METHODS = {
    "midpoint": {
        "A": np.array([[0.0, 0.0],
                       [0.5, 0.0]], dtype=float),
        "b": np.array([0.0, 1.0], dtype=float),
        "c": np.array([0.0, 0.5], dtype=float),
        "order": 2,
    },
    "heun": {
        "A": np.array([[0.0, 0.0],
                       [1.0, 0.0]], dtype=float),
        "b": np.array([0.5, 0.5], dtype=float),
        "c": np.array([0.0, 1.0], dtype=float),
        "order": 2,
    },
    "rk3_heun": {
        "A": np.array([[0.0, 0.0, 0.0],
                       [1.0 / 3.0, 0.0, 0.0],
                       [0.0, 2.0 / 3.0, 0.0]], dtype=float),
        "b": np.array([1.0 / 4.0, 0.0, 3.0 / 4.0], dtype=float),
        "c": np.array([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=float),
        "order": 3,
    },
    "rk3_simpson": {
        "A": np.array([[0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0],
                       [-1.0, 2.0, 0.0]], dtype=float),
        "b": np.array([1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0], dtype=float),
        "c": np.array([0.0, 0.5, 1.0], dtype=float),
        "order": 3,
    },
    "rk4_classic": {
        "A": np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0, 0.0],
                       [0.0, 0.5, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0]], dtype=float),
        "b": np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=float),
        "c": np.array([0.0, 0.5, 0.5, 1.0], dtype=float),
        "order": 4,
    },
    "rk4_38": {
        "A": np.array([[0.0, 0.0, 0.0, 0.0],
                       [1.0 / 3.0, 0.0, 0.0, 0.0],
                       [-1.0 / 3.0, 1.0, 0.0, 0.0],
                       [1.0, -1.0, 1.0, 0.0]], dtype=float),
        "b": np.array([1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0], dtype=float),
        "c": np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=float),
        "order": 4,
    },
}

"""
Функции rk_step, solve_ivp_fixed и solve_ivp_adaptive принимают аргумент method,
 который может быть любым из этих ключей, и автоматически используют соответствующие коэффициенты
"""

# Словарь для удобного обращения по номерам (26–31) или коротким именам
_ALIASES = {
    "26": "midpoint",
    "27": "heun",
    "28": "rk3_heun",
    "29": "rk3_simpson",
    "30": "rk4_classic",
    "31": "rk4_38",
    "mid": "midpoint",
    "midpoint": "midpoint",
    "heun": "heun",
    "trapezoid": "heun",
    "rk3_heun": "rk3_heun",
    "heun3": "rk3_heun",
    "rk3_simpson": "rk3_simpson",
    "simpson3": "rk3_simpson",
    "rk4_classic": "rk4_classic",
    "classic": "rk4_classic",
    "rk4_38": "rk4_38",
    "three_eighths": "rk4_38",
}

def _method_data(method):
    key = str(method).strip().lower()  # Приводим к строке, убираем пробелы, переводим в нижний регистр
    key = _ALIASES.get(key, key)  # Если key есть в словаре псевдонимов, берём имя метода, иначе оставляем как есть
    if key not in _METHODS:
        raise ValueError(
            "Unknown method. Use 26-31 or one of: midpoint, heun, rk3_heun, rk3_simpson, rk4_classic, rk4_38."
        )
    return key, _METHODS[key] # Возвращаем имя метода и его данные (матрицы A, b, c и порядок)

# Приводит начальное условие y0 к единому формату (одномерному массиву float)
def _as_state(y):
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0: # скаляр преобразуется в массив с одним элементом
        return arr.reshape(1), True # возвращает массив и флаг, указывающий, является ли y0 скаляром
    return arr.reshape(-1), False  # вектор или матрица преобразуется в одномерный массив

# Восстанавливает исходный формат y0
def _restore_state(y, scalar):
    y = np.asarray(y, dtype=float)
    return float(y[0]) if scalar else y

# Обёртка вокруг функции правой части
def _call_f(f, x, y):
    arg = float(y[0]) if y.size == 1 else y
    out = np.asarray(f(float(x), arg), dtype=float) # Вызываем f, результат – массив float

    if out.ndim == 0: # Если скаляр
        out = out.reshape(1) # Превращаем в массив из одного элемента
    else:
        out = out.reshape(-1) # Принудительно делаем его одномерным

    if out.size == 1 and y.size > 1: # Если f вернула скаляр, а система векторная
        out = np.full(y.shape, float(out[0]), dtype=float) # Размножаем скаляр на все компоненты

    return out  # Возвращаем корректный одномерный массив

# Вычисляет бесконечную норму вектора
def _norm_inf(v):
    v = np.asarray(v, dtype=float).reshape(-1) # Приводим входные данные к одномерному массиву float
    return float(np.max(np.abs(v))) if v.size else 0.0 # Если массив не пуст – макс. модуль, иначе 0.0

# Функция rk_step выполняет один шаг численного интегрирования ОДУ
def rk_step(f, x, y, h, method="rk4_classic"):
    """
    k_i = f( x + c_i·h ,  y + h·Σ_{j=1}^{i-1} a_{ij}·k_j ),   i = 1..s
    y_next = y + h·Σ_{i=1}^{s} b_i·k_i
    """

    _, tab = _method_data(method) # Получаем таблицу Бутчера (A, b, c, order)
    y_vec, scalar = _as_state(y)  # Приводим y к 1D массиву и запоминаем, был ли скаляром

    s = len(tab["b"]) # Количество стадий
    k = np.zeros((s, y_vec.size), dtype=float) # Массив для хранения векторов k_i

    for i in range(s): # Цикл по стадиям
        stage = y_vec.copy() # Начинаем с текущего значения y
        if i:
            # Вычисляем h * Σ(A[i][j] * k_j)
            stage += h * np.sum(tab["A"][i, :i, None] * k[:i], axis=0)  # Для i>0 добавляем вклад предыдущих стадий
        # Вычисляем k_i = f(x + c_i*h, stage)
        k[i] = _call_f(f, x + tab["c"][i] * h, stage)

    # Формируем новое значение: y_next = y + h * Σ(b_i * k_i)
    y_next = y_vec + h * np.sum(tab["b"][:, None] * k, axis=0)
    return _restore_state(y_next, scalar) # Возвращаем в исходной форме

# Интегрирование ОДУ на равномерной сетке фиксированным методом Рунге-Кутты
# последовательно применяет rk_step n_steps раз
def solve_ivp_fixed(f, x0, xf, y0, n_steps, method="rk4_classic"):
    # Проверки корректности входных данных
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if xf <= x0:
        raise ValueError("Require xf > x0.")

    name, tab = _method_data(method) # Получаем таблицу Бутчера и имя метода
    y, scalar = _as_state(y0) # Приводим y0 к 1D массиву, запоминаем флаг скаляра

    x = float(x0) # Текущее значение x
    h = (float(xf) - float(x0)) / n_steps # Постоянный шаг сетки

    xs = [x] # Список для сохранения узлов x
    ys = [y.copy()] # Список для сохранения решений y
    rhs_calls = 0 # Счётчик вызовов правой части f

    for _ in range(n_steps): # Цикл по шагам интегрирования
        s = len(tab["b"]) # Количество стадий
        k = np.zeros((s, y.size), dtype=float) # Массив для хранения стадий k_i

        # Вычисление всех стадий (как в rk_step)
        for i in range(s):
            stage = y.copy() # Начальное приближение стадии
            if i:
                # Добавляем вклад предыдущих стадий: h * Σ(A[i][j] * k_j)
                stage += h * np.sum(tab["A"][i, :i, None] * k[:i], axis=0)
            k[i] = _call_f(f, x + tab["c"][i] * h, stage)  # Вычисляем k_i

        rhs_calls += s # Учитываем все вызовы f на этом шаге
        # y_{next} = y + h * Σ(b_i * k_i)
        y = y + h * np.sum(tab["b"][:, None] * k, axis=0)
        x += h # Переходим к следующему узлу

        xs.append(x) # Сохраняем x
        ys.append(y.copy()) # Сохраняем y

    # Преобразуем историю в массивы numpy
    x_hist = np.asarray(xs, dtype=float)
    y_hist = np.asarray(ys, dtype=float)
    if scalar: # Если исходная задача была скалярной
        y_hist = y_hist[:, 0] # Извлекаем единственный столбец

    # Возвращаем словарь с результатами
    return {
        "x": x_hist,
        "y": y_hist,
        "method": name,
        "order": tab["order"],
        "n_steps": n_steps,
        "rhs_calls": rhs_calls,
    }

# Оценка начального шага для адаптивного интегрирования
def estimate_initial_step(f, x0, xf, y0, method="rk4_classic", tol=1e-6, max_step=None):
    _, tab = _method_data(method) # Получаем данные метода
    y_vec, _ = _as_state(y0) # Приводим y0 к 1D массиву
    f0 = _call_f(f, float(x0), y_vec) # Вычисляем f(x0, y0) – производную в начальной точке

    interval = abs(float(xf) - float(x0)) # Длина интервала интегрирования
    if interval == 0:
        raise ValueError("xf must differ from x0.")

    # Масштаб производной (не даём стать нулём, чтобы избежать деления на ноль)
    scale = max(_norm_inf(f0), 1e-14) # Бесконечная норма f0, но не менее 1e-14

    # Основная формула: h = (tol / scale)^{1/(p+1)}
    h = (tol / scale) ** (1.0 / (tab["order"] + 1))

    # Если получилось нечисловое или неположительное значение – берём запасной шаг
    if not np.isfinite(h) or h <= 0:
        h = interval / 100.0

    # Ограничиваем сверху, если задан max_step
    if max_step is not None:
        h = min(h, float(max_step))

    # Шаг не может быть больше длины всего интервала
    return min(h, interval)

# Адаптивное интегрирование ОДУ с контролем шага методом удвоения
# интегрирование с автоматическим выбором шага, чтобы локальная ошибка на каждом шаге не превышала заданных допусков rtol (относительный) и atol (абсолютный).
def solve_ivp_adaptive(
    f,
    x0,
    xf,
    y0,
    method="rk4_classic",
    rtol=1e-6,
    atol=1e-12,
    h0=None,
    h_min=1e-14,
    h_max=None,
    safety=0.9,
    max_steps=100000,
):
    #   rtol - относительный допуск локальной ошибки (>0)
    #   atol - абсолютный допуск локальной ошибки (>=0)
    #   h0 - начальный шаг
    #   h_min - минимально допустимый шаг
    #   h_max - максимально допустимый шаг (None - интервал)
    # safety - коэффициент запаса (0< safety <1), уменьшает шаг для снижения числа отвержений


    # Проверки корректности входных данных
    if xf <= x0:
        raise ValueError("Require xf > x0.")
    if rtol <= 0 or atol < 0:
        raise ValueError("rtol must be > 0 and atol must be >= 0.")

    name, tab = _method_data(method) # Данные метода
    y, scalar = _as_state(y0) # y – 1D массив, scalar – флаг
    x = float(x0)
    interval = float(xf - x0)

    # Начальный шаг либо переданный, либо оценённый
    if h0 is None:
        h = estimate_initial_step(f, x0, xf, y0, method=name, tol=max(rtol, atol), max_step=interval)
    else:
        h = float(h0)

    if h_max is None:
        h_max = interval

    h = min(max(h, h_min), h_max, interval) # Принудительные ограничения

    # История решения
    xs = [x]
    ys = [y.copy()]
    accepted_steps = [] # Храним все принятые шаги
    rejected_steps = [] # Храним все отвергнутые шаги
    rhs_calls = 0 # Количество вызванных f
    n_accept = 0 # Количество принятых шагов
    n_reject = 0 # Количество отвергнутых шагов

    p = tab["order"] # Порядок точности метода
    corr = 2**p - 1 # Поправочный коэффициент для экстраполяции Ричардсона

    # Основной цикл адаптивного интегрирования
    while x < xf and (n_accept + n_reject) < max_steps:
        h = min(h, xf - x) # Не перескакиваем конечную точку
        if h < h_min:
            raise RuntimeError("Step size underflow.")

        # Вычисляем два приближения:
        # y_full – один большой шаг h
        # y_two_half – два маленьких шага h/2
        y_full = np.asarray(rk_step(f, x, y, h, method=name), dtype=float).reshape(-1)
        y_half = np.asarray(rk_step(f, x, y, h / 2.0, method=name), dtype=float).reshape(-1)
        y_two_half = np.asarray(rk_step(f, x + h / 2.0, y_half, h / 2.0, method=name), dtype=float).reshape(-1)
        rhs_calls += 3 * len(tab["b"]) # 3 набора стадий (один шаг h и два по h/2)

        # Оценка локальной ошибки
        diff = y_two_half - y_full # Разность между двумя способами
        scale = atol + rtol * np.maximum(np.abs(y_full), np.abs(y_two_half)) # Масштаб для нормировки
        err = _norm_inf(diff / scale) # Относительная ошибка (в бесконечной норме)
        accepted = err <= 1.0 # Принят ли шаг

        # Вычисляем коэффициент изменения шага (стандартная формула для метода удвоения)
        if err == 0.0:
            factor = 2.0
        else:
            factor = safety * err ** (-1.0 / (p + 1))
        factor = float(np.clip(factor, 0.2, 5.0)) # Ограничиваем, чтобы шаг не менялся слишком резко

        if accepted:
            # Уточняем решение по Ричардсону (экстраполяция)
            """
            Экстраполяция Ричардсона — это метод повышения точности численного результата путём комбинирования двух приближений, полученных с разными шагами.
            """
            y = y_two_half + diff / corr
            x += h
            xs.append(x)
            ys.append(y.copy())
            accepted_steps.append(h)
            n_accept += 1
            h = min(h * factor, h_max) # Увеличиваем шаг для следующей итерации
        else:
            rejected_steps.append(h)
            n_reject += 1
            # Уменьшаем шаг, но не более чем до h_min
            h = max(h * max(0.2, min(0.8, factor)), h_min)

    if x < xf:
        raise RuntimeError("Maximum number of steps exceeded before reaching xf.")

    # Преобразуем историю в массивы numpy
    x_hist = np.asarray(xs, dtype=float)
    y_hist = np.asarray(ys, dtype=float)
    if scalar:
        y_hist = y_hist[:, 0]

    # Возвращаем подробный словарь результатов
    return {
        "x": x_hist,
        "y": y_hist,
        "method": name,
        "order": p,
        "accepted_steps": np.asarray(accepted_steps, dtype=float),
        "rejected_steps": np.asarray(rejected_steps, dtype=float),
        "rhs_calls": rhs_calls,
        "iterations": n_accept + n_reject,
        "accepted_count": n_accept,
        "rejected_count": n_reject,
        "rtol": rtol,
        "atol": atol,
    }
