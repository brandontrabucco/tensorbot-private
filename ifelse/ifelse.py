import numpy as np

def pt(x, y):
    return np.array([x, y])

class Agent:
    field_dimensions = (15, 15)
    field_obstacles = [pt(5, 5), pt(10, 10)]
    buff_zone = pt(1, 1)
    reload_zone = pt(14, 14)
    danger_samples = 50
    length_factor = .1
    init_boost = 10
    boost_multiplier = .8
    weights = np.ones(6)

    def __init__(self, p, covp, ep1, covep1, ep2, covep2, ammo, health, can_activate):
        self.p = p
        self.covp = covp
        self.ep1 = ep1
        self.covep1 = covep1
        self.ep2 = ep2
        self.covep2 = covep2
        self.ammo = ammo
        self.health = health
        self.curstate = -1
        self.stay = 1
        self.can_activate = can_activate

    def step(self, state, p, covp, ep1, covep1, ep2, covep2, ammo, health, can_activate):
        self.p = p
        self.covp = covp
        self.ep1 = ep1
        self.covep1 = covep1
        self.ep2 = ep2
        self.covep2 = covep2
        self.ammo = ammo
        self.health = health
        self.can_activate = can_activate

        if state == self.curstate:
            self.stay *= Agent.boost_multiplier
        else:
            self.stay = Agent.init_boost



    def get_vector(self):
        vec = np.array([self.defend_score(), self.buff_score(), self.reload_score(),
                        self.search_score(), self.attack_score(), self.escape_score()])
        vec = vec * self.weights
        if self.curstate >= 0:
            return vec * np.array([1 if i != self.curstate else self.stay for i in range(6)])
        return vec

    def defend_score(self):
        return 1/self.curr_danger_score()

    def buff_score(self):
        if self.can_activate:
            return 1/self.danger_score_path(Agent.buff_zone)
        return 0

    def reload_score(self):
        return 1/self.ammo/self.danger_score_path(Agent.reload_zone)

    def search_score(self):
        return min(1/np.linalg.norm(self.covep1), 1/np.linalg.norm(self.covep2))

    def attack_score(self):
        return self.ammo/min(np.linalg.norm(self.covep1)*np.linalg.norm(self.p - self.ep1), np.linalg.norm(self.covep2)*np.linalg.norm(self.p - self.ep2))

    def escape_score(self):
        return self.curr_danger_score()/self.health

    def sample_location(self):
        return np.random.multivariate_normal(self.p, self.covp)

    def sample_enemies(self):
        return np.random.multivariate_normal(
            self.ep1, self.covep1), np.random.multivariate_normal(self.ep2, self.covep2)

    def curr_danger_score(self):
        out = 0
        for i in range(Agent.danger_samples):
            out += self.danger_points(self.sample_location(), *self.sample_enemies())
        return out/Agent.danger_samples

    def danger_score_path(self, p):
        score = 0
        for i in range(Agent.danger_samples):
            score += self.danger_points(
                linterp(self.sample_location(), p, i/(Agent.danger_samples - 1)), *self.sample_enemies())
        return (score/Agent.danger_samples + Agent.length_factor)*np.linalg.norm(p - self.p)

    def danger_points(self, p, e1, e2):
        ret = 0
        if all([not line_intersect_points((p, e1), l) for l in Agent.field_obstacles]):
            ret += 1/np.linalg.norm(e1-p)

        if all([not line_intersect_points((p, e2), l) for l in Agent.field_obstacles]):
            ret += 1/np.linalg.norm(e2-p)

        return ret


def line_from_points(p1, p2):
    return np.linalg.solve(np.array([[p1[0], p1[1], -1], [p2[0], p2[1], -1], [1, 1, 1]]), np.array([0, 0, 1]))


def line_intersect_points(l1, l2):
    mat = np.array(line_from_points(*l1), line_from_points(*l2))
    a = mat[:, :-1]
    b = mat[:, -1]
    p = np.linalg.solve(a, b)
    return (p[0] - l1[0][0])*(p[0] - l1[1][0]) < 0 and (p[0] - l2[0][0])*(p[0] - l2[1][0])


def linterp(p1, p2, x):
    return p1*(1-x) + p2