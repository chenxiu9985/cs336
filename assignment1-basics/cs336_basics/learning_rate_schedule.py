import math

def cosine_annealing(t, alpha_max, alpha_min, T_w, T_c):
    if not (0 <= T_w < T_c):
        raise ValueError("need require 0 <= T_w < T_c")
    if not (0 <= alpha_min <= alpha_max):
        raise ValueError("need require 0 <= alpha_min <= alpha_max")
    if t < 0:
        raise ValueError("t is not negative number")

    if t < T_w:
        return  t / T_w * alpha_max
    elif t >= T_w and t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (1 + math.cos(progress*math.pi)) * (alpha_max-alpha_min)
    else:
        return alpha_min