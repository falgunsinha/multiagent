import sys
import time
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# =========================
# Environment (same logic as before, returns reward components in info)
# =========================
class GridCollectEnv(gym.Env):
    """
    6x6 Grid with random items (5–8). Agent moves and PICKs to clear items.
    Reward shaping:
      r_total = r_dist + r_pick + r_clear + r_invalid
      r_dist   = (prev_dist - new_dist) toward origin at (gs, gs) outside bottom-right
      r_pick   = +pick_reward on item
      r_clear  = +clear_bonus on last item
      r_invalid= -0.02 when PICK on empty cell
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size=6,
        min_items=5,
        max_items=8,
        max_steps=None,
        seed=None,
        distance_reward=True,
        pick_reward=1.0,
        clear_bonus=2.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.min_items = min_items
        self.max_items = max_items
        self.max_steps = max_steps or (grid_size * grid_size * 2)
        self._rng = np.random.default_rng(seed)

        self.distance_reward = distance_reward
        self.pick_reward = pick_reward
        self.clear_bonus = clear_bonus

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2, grid_size, grid_size), dtype=np.float32
        )

        self.agent_pos = None
        self.items = None
        self.steps = 0
        self.origin_rc = (self.grid_size, self.grid_size)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.items = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        n_items = self._rng.integers(self.min_items, self.max_items + 1)
        indices = self._rng.choice(self.grid_size ** 2, size=n_items, replace=False)
        self.items.flat[indices] = 1

        # agent on empty tile
        while True:
            r, c = self._rng.integers(0, self.grid_size, size=2)
            if self.items[r, c] == 0:
                self.agent_pos = (r, c)
                break
        return self._get_obs(), {}

    def _euclid_distance_cells(self, rc_a, rc_b):
        (ra, ca), (rb, cb) = rc_a, rc_b
        ax, ay = ca + 0.5, ra + 0.5
        bx, by = cb + 0.5, rb + 0.5
        return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))

    def step(self, action):
        assert self.action_space.contains(action)
        self.steps += 1

        prev_dist = self._euclid_distance_cells(self.agent_pos, self.origin_rc)

        r_total = 0.0
        r_dist = 0.0
        r_pick = 0.0
        r_clear = 0.0
        r_invalid = 0.0
        terminated = False

        r, c = self.agent_pos
        if action == 0 and r > 0:                         # UP
            r -= 1
        elif action == 1 and r < self.grid_size - 1:      # DOWN
            r += 1
        elif action == 2 and c > 0:                       # LEFT
            c -= 1
        elif action == 3 and c < self.grid_size - 1:      # RIGHT
            c += 1
        elif action == 4:                                 # PICK
            if self.items[r, c] == 1:
                self.items[r, c] = 0
                r_pick += self.pick_reward
            else:
                r_invalid += -0.02

        self.agent_pos = (r, c)
        new_dist = self._euclid_distance_cells(self.agent_pos, self.origin_rc)
        r_dist += (prev_dist - new_dist)

        if not self.items.any():
            r_clear += self.clear_bonus
            terminated = True

        r_total = r_dist + r_pick + r_clear + r_invalid
        info = {
            "r_total": r_total,
            "r_dist": r_dist,
            "r_pick": r_pick,
            "r_clear": r_clear,
            "r_invalid": r_invalid,
            "dist_to_origin": new_dist,
        }

        if self.steps >= self.max_steps:
            terminated = True

        return self._get_obs(), r_total, terminated, False, info

    def _get_obs(self):
        obs = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        ar, ac = self.agent_pos
        obs[0, ar, ac] = 1.0
        obs[1] = self.items.astype(np.float32)
        return obs


# =========================
# Pygame — resizable window & responsive scaling
# =========================
# Colors
BG_COLOR      = (24, 24, 24)
GRID_COLOR    = (60, 60, 60)
OUTER_COLOR   = (35, 35, 35)
CIRCLE_COLOR  = (235, 64, 52)
TEXT_COLOR    = (220, 220, 220)
ACCENT        = (160, 200, 255)
AGENT_RGBA    = (63, 127, 255, 150)
AGENT_BORDER  = (130, 170, 255)
ORIGIN_COLOR  = (255, 212, 64)

# Controls
HELP_LINES = [
    "Controls:",
    "Arrows / WASD = Move",
    "Space = PICK",
    "N = New episode",
    "R = Reset same seed",
    "Esc / Q = Quit",
    "Resize the window freely",
]

def compute_layout(win_w, win_h, grid_size, sidebar_min=300, sidebar_frac=0.28):
    """
    Given the current window size, compute cell size, sidebar width, and outer band size,
    keeping aspect and providing a visible band outside bottom-right for the origin.
    """
    # Sidebar width adapts but never below sidebar_min
    sidebar_w = max(sidebar_min, int(win_w * sidebar_frac))

    # We'll compute CELL to fit the grid + outer band inside remaining width/height
    # Let OUTER_BAND be proportional to CELL so it scales gracefully
    # Solve iteratively: guess CELL, compute band, check fit, adjust.
    # Simple approach: start with height constraint, then clamp to width constraint.
    # Height constraint: gs*CELL + band <= win_h  with band = 0.5*CELL  -> CELL*(gs+0.5) <= win_h
    cell_h = max(20, (win_h // (grid_size + 1)))  # rough first pass
    # Width constraint: gs*CELL + band + sidebar <= win_w
    cell_w = max(20, ((win_w - sidebar_w) // (grid_size + 1)))

    CELL = max(20, min(cell_h, cell_w))  # final CELL
    OUTER_BAND = max(16, int(0.6 * CELL))  # a bit thicker so origin is always clear

    # Re-check with band included
    max_cell_h = (win_h - OUTER_BAND) // grid_size
    max_cell_w = (win_w - sidebar_w - OUTER_BAND) // grid_size
    CELL = max(20, min(CELL, max_cell_h, max_cell_w))
    OUTER_BAND = max(16, int(0.6 * CELL))

    grid_w = grid_size * CELL
    grid_h = grid_size * CELL
    board_w = grid_w + OUTER_BAND
    board_h = grid_h + OUTER_BAND

    return CELL, OUTER_BAND, sidebar_w, grid_w, grid_h, board_w, board_h

def make_fonts(base_cell):
    # Scale fonts with cell size for readability
    size_main = max(14, min(24, base_cell // 3))
    size_small = max(12, min(20, base_cell // 4))
    return (pygame.font.SysFont("consolas", size_main),
            pygame.font.SysFont("consolas", size_small))

def draw_origin(surface, grid_w, grid_h, OUTER_BAND, CELL):
    # Center origin in the outer band corner
    cx = grid_w + OUTER_BAND // 2
    cy = grid_h + OUTER_BAND // 2
    radius = max(12, CELL // 3)
    ring_w = max(2, radius // 6)
    line_w = max(3, radius // 5)

    pygame.draw.line(surface, ORIGIN_COLOR, (cx - radius - 8, cy), (cx + radius + 8, cy), line_w)
    pygame.draw.line(surface, ORIGIN_COLOR, (cx, cy - radius - 8), (cx, cy + radius + 8), line_w)
    pygame.draw.circle(surface, ORIGIN_COLOR, (cx, cy), max(4, radius // 3))
    pygame.draw.circle(surface, ORIGIN_COLOR, (cx, cy), radius, ring_w)

def draw_board(surface, env, CELL, OUTER_BAND, grid_w, grid_h):
    # Background
    surface.fill(BG_COLOR)

    # Outer band (right and bottom)
    pygame.draw.rect(surface, OUTER_COLOR, (grid_w, 0, OUTER_BAND, grid_h + OUTER_BAND))
    pygame.draw.rect(surface, OUTER_COLOR, (0, grid_h, grid_w, OUTER_BAND))

    # Grid lines
    for i in range(env.grid_size + 1):
        x = i * CELL
        y = i * CELL
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, grid_h), 1)
        pygame.draw.line(surface, GRID_COLOR, (0, y), (grid_w, y), 1)

    # Items (red circles)
    ys, xs = np.where(env.items == 1)
    for r, c in zip(ys, xs):
        cx = c * CELL + CELL // 2
        cy = r * CELL + CELL // 2
        radius = max(6, CELL // 3)
        pygame.draw.circle(surface, CIRCLE_COLOR, (cx, cy), radius)

    # Agent (semi-transparent rounded box)
    ar, ac = env.agent_pos
    box_w = max(6, int(CELL * 0.86))
    box_h = max(6, int(CELL * 0.86))
    box_x = ac * CELL + (CELL - box_w) // 2
    box_y = ar * CELL + (CELL - box_h) // 2

    agent_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    pygame.draw.rect(agent_surf, AGENT_RGBA, agent_surf.get_rect(), border_radius=int(CELL * 0.18))
    surface.blit(agent_surf, (box_x, box_y))
    pygame.draw.rect(surface, AGENT_BORDER, (box_x, box_y, box_w, box_h), width=max(1, CELL // 20), border_radius=int(CELL * 0.18))

def blit_sidebar(surface, env, episode, steps, totals, done_flag, fps_val, fonts, board_w, board_h, sidebar_w):
    font, smallfont = fonts
    x0 = board_w + 16
    y = 10

    def line(text, color=TEXT_COLOR, f=font, pad=4):
        nonlocal y
        surf = f.render(text, True, color)
        surface.blit(surf, (x0, y))
        y += surf.get_height() + pad

    def small(text, color=TEXT_COLOR, pad=2):
        line(text, color=color, f=smallfont, pad=pad)

    # Header
    line("GridCollect — Resizable", ACCENT)
    line(f"Episode: {episode}")
    line(f"Steps:   {steps}")
    y += 4

    # Totals
    line("Cumulative Rewards", ACCENT)
    line(f"Total:  {totals['total']:.3f}")
    small(f"  Distance: {totals['dist']:.3f}")
    small(f"  Picks:    {totals['pick']:.3f}")
    small(f"  Clear:    {totals['clear']:.3f}")
    small(f"  Invalid:  {totals['invalid']:.3f}")
    y += 6

    # Live metrics
    dist = env._euclid_distance_cells(env.agent_pos, env.origin_rc)
    line(f"Items left: {int(env.items.sum())}")
    line(f"Dist→Origin: {dist:.3f}")
    line(f"Done: {done_flag}")
    line(f"FPS:  {fps_val:.1f}")
    y += 6

    # Reward docs
    line("Reward Function", ACCENT)
    small("r_total = r_dist + r_pick + r_clear + r_invalid")
    small("r_dist   = (prev_dist - new_dist)")
    small(f"r_pick   = +{env.pick_reward:.2f} if on a circle")
    small(f"r_clear  = +{env.clear_bonus:.2f} when last circle removed")
    small("r_invalid= -0.02 when PICK on empty")
    small("Origin at (gs, gs) just outside bottom-right")
    y += 6

    # Controls
    line("Controls", ACCENT)
    for s in HELP_LINES:
        small(s)

def main():
    # --- config ---
    grid_size = 6
    min_items, max_items = 27, 36
    max_steps = 120
    seed = int(time.time())

    env = GridCollectEnv(
        grid_size=grid_size, min_items=min_items, max_items=max_items,
        max_steps=max_steps, seed=seed, distance_reward=True,
        pick_reward=1.0, clear_bonus=2.0
    )
    obs, _ = env.reset()

    pygame.init()
    # Start with a comfortable default size; make it resizable
    start_w, start_h = 1200, 800
    screen = pygame.display.set_mode((start_w, start_h), pygame.RESIZABLE)
    pygame.display.set_caption("GridCollectEnv — Manual Test (Resizable)")

    # State/UI
    episode = 1
    steps = 0
    done_flag = False
    totals = dict(total=0.0, dist=0.0, pick=0.0, clear=0.0, invalid=0.0)

    def add_components(info):
        totals["total"] += info.get("r_total", 0.0)
        totals["dist"] += info.get("r_dist", 0.0)
        totals["pick"] += info.get("r_pick", 0.0)
        totals["clear"] += info.get("r_clear", 0.0)
        totals["invalid"] += info.get("r_invalid", 0.0)

    def reset_cumulatives():
        totals.update(total=0.0, dist=0.0, pick=0.0, clear=0.0, invalid=0.0)

    def take_action(action):
        nonlocal steps, done_flag, obs
        obs, reward, terminated, _, info = env.step(action)
        steps += 1
        done_flag = terminated
        add_components(info)

    clock = pygame.time.Clock()

    running = True
    while running:
        dt = clock.tick(60)
        fps_val = clock.get_fps()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                # Recreate the window at the new size; keep RESIZABLE flag
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k in (pygame.K_UP, pygame.K_w):
                    take_action(0)
                elif k in (pygame.K_DOWN, pygame.K_s):
                    take_action(1)
                elif k in (pygame.K_LEFT, pygame.K_a):
                    take_action(2)
                elif k in (pygame.K_RIGHT, pygame.K_d):
                    take_action(3)
                elif k == pygame.K_SPACE:
                    take_action(4)
                elif k == pygame.K_n:
                    obs, _ = env.reset()
                    episode += 1
                    steps = 0
                    done_flag = False
                    reset_cumulatives()
                elif k == pygame.K_r:
                    obs, _ = env.reset()
                    steps = 0
                    done_flag = False
                    reset_cumulatives()

        # Compute responsive layout from current window size
        win_w, win_h = screen.get_size()
        CELL, OUTER_BAND, sidebar_w, grid_w, grid_h, board_w, board_h = compute_layout(win_w, win_h, env.grid_size)
        fonts = make_fonts(CELL)

        # Draw board to a sub-surface
        # Left area: board (grid + outer band), Right area: sidebar
        board_surface = screen.subsurface(pygame.Rect(0, 0, board_w, board_h))
        draw_board(board_surface, env, CELL, OUTER_BAND, grid_w, grid_h)
        draw_origin(board_surface, grid_w, grid_h, OUTER_BAND, CELL)

        # Sidebar surface (remaining right area)
        sidebar_rect = pygame.Rect(board_w, 0, max(0, win_w - board_w), win_h)
        sidebar_surface = screen.subsurface(sidebar_rect)
        sidebar_surface.fill(BG_COLOR)
        blit_sidebar(sidebar_surface, env, episode, steps, totals, done_flag, fps_val, fonts, board_w=0, board_h=0, sidebar_w=sidebar_rect.width)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
