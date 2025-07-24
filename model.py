import numpy as np

class ASPStackelbergAnalytical:
    """
    Analytical solver for Stackelberg equilibrium trong Stackelberg Game:
    Tính giá p*, utility tối ưu của ASP, và utility từng VMU analytic.
    """
    def __init__(self, env):
        # Lấy các tham số mô hình từ env (chuẩn hóa theo code env.py bạn đã dùng)
        self.K = env.num_users if hasattr(env, "num_users") else env.num_vmu
        self.alpha = env.alpha
        self.beta = env.beta
        self.gamma = env.gamma
        self.c = env.cost if hasattr(env, "cost") else env.cost_per_step
        self.s_min = env.s_min if hasattr(env, "s_min") else 120
        self.s_max = env.s_max if hasattr(env, "s_max") else 160
        # Tham số delta_k (hệ số hài lòng từng user), lấy từ env hoặc mặc định
        if hasattr(env, "delta_k"):
            self.delta_k = np.array(env.delta_k)
        else:
            self.delta_k = np.ones(self.K) * 200

    def best_response_sk(self, p, delta_k):
        """Tính analytic S_k* cho mỗi user"""
        if self.alpha == 0:
            return self.s_max
        s_star = (p / (2 * delta_k * self.alpha)) - (self.beta / (2 * self.alpha))
        return np.clip(s_star, self.s_min, self.s_max)

    def Up(self, p):
        """
        Lợi ích ASP analytic với giá p
        """
        S_star = np.array([self.best_response_sk(p, dk) for dk in self.delta_k])
        # Chỉ giữ những user thực sự tham gia (utility >= 0)
        SSIM = self.alpha * S_star**2 + self.beta * S_star + self.gamma
        Uk = self.delta_k * SSIM - p * S_star
        mask = Uk >= 0
        return np.sum((p - self.c) * S_star[mask]), S_star, mask

    def search(self, p_min=1, p_max=20, num_grid=1000):
        """
        Tìm p* analytic, utility tối ưu của ASP, trả về: (p*, S*_all, Up*)
        """
        p_grid = np.linspace(p_min, p_max, num_grid)
        utils = []
        for p in p_grid:
            util, _, _ = self.Up(p)
            utils.append(util)
        idx = np.argmax(utils)
        p_star = p_grid[idx]
        Up_star, S_star, mask = self.Up(p_star)
        return p_star, S_star, Up_star
