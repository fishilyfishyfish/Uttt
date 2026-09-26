"""궁극의 틱택토(Ultimate Tic-Tac-Toe)를 알파제로 방식으로 학습시킨다.

이 파일 하나에 규칙, 신경망, 탐색, 학습, 성적표가 다 들어 있다.
    python uttt.py --config config.json

규칙
  3x3 판 안에 3x3 판이 또 있다. 칸은 모두 81개.
  작은 판은 보통 틱택토처럼 3개를 이으면 먹는다.
  작은 판 3개를 한 줄로 먹으면 최종 승리.
  내가 둔 '작은 판 안에서의 위치'가 상대가 둬야 할 작은 판을 정한다.
  보내진 작은 판이 이미 끝났으면 상대는 아무 데나 둘 수 있다.
"""
from __future__ import annotations

import argparse, json, math, os, time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp

STATE_DIR = "state"
FULL9 = 0x1FF
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
LINE_MASKS = [(1 << a) | (1 << b) | (1 << c) for a, b, c in LINES]
BOARD_MASK = [FULL9 << (b * 9) for b in range(9)]


# ============================================================ 규칙
def start_state():
    """(내 돌, 상대 돌, 내가 먹은 작은판, 상대가 먹은 작은판, 비긴 작은판, 보내진 판)
    항상 '둘 차례인 사람' 기준으로 뒤집어 들고 다닌다."""
    return (0, 0, 0, 0, 0, -1)


def decided(s):
    return s[2] | s[3] | s[4]


def legal_moves(s):
    me, op, _, _, _, forced = s
    occ = me | op
    dec = decided(s)
    if forced >= 0 and not (dec >> forced) & 1:
        boards = [forced]
    else:
        boards = [b for b in range(9) if not (dec >> b) & 1]
    out = []
    for b in boards:
        free = (~occ >> (b * 9)) & FULL9
        base = b * 9
        while free:
            i = (free & -free).bit_length() - 1
            out.append(base + i)
            free &= free - 1
    return out


def won_small(bits9):
    return any(bits9 & m == m for m in LINE_MASKS)


def apply_move(s, cell):
    """수를 둔 뒤의 상태를 '다음에 둘 사람' 기준으로 뒤집어서 돌려준다.
    반환: (새 상태, 끝났는지, 결과)  결과는 방금 둔 사람 기준 +1 승 / 0 무 / None 진행중"""
    me, op, wm, wo, wd, _ = s
    me |= 1 << cell
    b, i = divmod(cell, 9)

    mine9 = (me >> (b * 9)) & FULL9
    if won_small(mine9):
        wm |= 1 << b
    elif ((me | op) >> (b * 9)) & FULL9 == FULL9:
        wd |= 1 << b

    ns = (op, me, wo, wm, wd, i if not ((wm | wo | wd) >> i) & 1 else -1)

    if won_small(wm):
        return ns, True, 1                      # 방금 둔 사람 승
    if (wm | wo | wd) == FULL9:                  # 큰 판이 다 찼다
        if TIEBREAK_BY_COUNT:
            a, c = bin(wm).count("1"), bin(wo).count("1")
            return ns, True, 1 if a > c else (0 if a == c else -1)
        return ns, True, 0
    return ns, False, None


TIEBREAK_BY_COUNT = True   # 큰 판이 다 찼는데 세 줄이 없으면 작은 판을 더 많이 먹은 쪽 승


def render(s, turn_mark="X"):
    me, op, wm, wo, wd, forced = s
    g = [["." for _ in range(9)] for _ in range(9)]
    for c in range(81):
        b, i = divmod(c, 9)
        r = (b // 3) * 3 + i // 3
        col = (b % 3) * 3 + i % 3
        if (me >> c) & 1:
            g[r][col] = turn_mark
        elif (op >> c) & 1:
            g[r][col] = "O" if turn_mark == "X" else "X"
    lines = []
    for r in range(9):
        row = " ".join("".join(g[r][k * 3:k * 3 + 3]) for k in range(3))
        lines.append(row)
        if r % 3 == 2 and r < 8:
            lines.append("-" * len(row))
    lines.append(f"보내진 판: {forced if forced >= 0 else '아무데나'}")
    return "\n".join(lines)


# ============================================================ 입력 변환
IN_DIM = 81 * 2 + 9 * 3 + 10


def features(states):
    n = len(states)
    x = np.zeros((n, IN_DIM), dtype=np.float32)
    for k, (me, op, wm, wo, wd, forced) in enumerate(states):
        for c in range(81):
            if (me >> c) & 1:
                x[k, c] = 1.0
            elif (op >> c) & 1:
                x[k, 81 + c] = 1.0
        for b in range(9):
            if (wm >> b) & 1:
                x[k, 162 + b] = 1.0
            if (wo >> b) & 1:
                x[k, 171 + b] = 1.0
            if (wd >> b) & 1:
                x[k, 180 + b] = 1.0
        if forced >= 0:
            x[k, 189 + forced] = 1.0
        else:
            x[k, 198] = 1.0
    return x


class Net(nn.Module):
    def __init__(self, hidden=256, layers=3):
        super().__init__()
        body = [nn.Linear(IN_DIM, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            body += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.body = nn.Sequential(*body)
        self.policy = nn.Linear(hidden, 81)
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)


class Evaluator:
    def __init__(self, net):
        self.net, self.cache = net, {}

    def __call__(self, s, legal):
        hit = self.cache.get(s)
        if hit is not None:
            return hit
        with torch.no_grad():
            logit, v = self.net(torch.from_numpy(features([s])))
        logit = logit[0].numpy()
        mask = np.full(81, -1e9, dtype=np.float32)
        mask[legal] = logit[legal]
        mask -= mask.max()
        p = np.exp(mask)
        p /= p.sum()
        out = (p, float(v[0]))
        self.cache[s] = out
        return out


# ============================================================ 탐색 (MCTS)
class Node:
    __slots__ = ("P", "N", "W", "acts")

    def __init__(self, P, acts):
        self.acts, self.P = acts, P
        self.N = np.zeros(len(acts), dtype=np.int32)
        self.W = np.zeros(len(acts), dtype=np.float64)


class MCTS:
    def __init__(self, ev, sims=400, c_puct=1.5, d_alpha=0.5, d_eps=0.25, rng=None):
        self.ev, self.sims, self.c = ev, sims, c_puct
        self.da, self.de = d_alpha, d_eps
        self.rng = rng or np.random.default_rng()
        self.tree = {}
        self._noise = None

    def reset(self):
        self.tree.clear()

    def _search(self, s, root_noise=False):
        node = self.tree.get(s)
        if node is None:
            acts = legal_moves(s)
            p, v = self.ev(s, acts)
            self.tree[s] = Node(np.array([p[a] for a in acts]), acts)
            return v

        P = node.P
        if root_noise and self._noise is not None and len(P) == len(self._noise):
            P = (1 - self.de) * P + self.de * self._noise

        tot = node.N.sum()
        q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
        i = int(np.argmax(q + self.c * P * math.sqrt(tot + 1e-8) / (1 + node.N)))
        a = node.acts[i]

        child, over, res = apply_move(s, a)
        # res는 '방금 둔 사람' 기준이고 지금 노드의 차례와 같은 사람이므로 그대로 쓴다.
        # 끝나지 않았으면 자식은 상대 차례이므로 부호를 뒤집는다.
        v = float(res) if over else -self._search(child)
        node.N[i] += 1
        node.W[i] += v
        return v

    def run(self, s, add_noise=True):
        # 탐험용 잡음은 한 수당 한 번만 뽑는다 (매 시뮬레이션마다 뽑으면 평균이 되어 효과가 사라진다)
        self._noise = None
        if add_noise:
            n = len(legal_moves(s))
            if n > 1:
                self._noise = self.rng.dirichlet([self.da] * n)
        for _ in range(self.sims):
            self._search(s, root_noise=add_noise)
        node = self.tree[s]
        pi = np.zeros(81)
        for i, a in enumerate(node.acts):
            pi[a] = node.N[i]
        t = pi.sum()
        return pi / t if t else pi


# ============================================================ 상대 선수들
def random_player(rng):
    return lambda s: int(rng.choice(legal_moves(s)))


def heuristic_player(rng):
    """작은 판을 딸 수 있으면 딴다. 상대가 딸 자리면 막는다. 아니면 무작위."""
    def play(s):
        acts = legal_moves(s)
        for a in acts:                                   # 내가 먹는 수
            _, over, res = apply_move(s, a)
            b = a // 9
            me2 = s[0] | (1 << a)
            if won_small((me2 >> (b * 9)) & FULL9) or (over and res == 1):
                return a
        blocks = []                                      # 상대가 먹을 자리 막기
        for a in acts:
            b = a // 9
            op2 = s[1] | (1 << a)
            if won_small((op2 >> (b * 9)) & FULL9):
                blocks.append(a)
        if blocks:
            return int(rng.choice(blocks))
        center = [a for a in acts if a % 9 == 4]          # 작은 판 가운데 선호
        return int(rng.choice(center if center else acts))
    return play


def net_player(net, sims, c_puct, rng, temp_moves=6):
    ev = Evaluator(net)
    box = {"ply": 0}

    def play(s):
        m = MCTS(ev, sims=sims, c_puct=c_puct, rng=rng)
        pi = m.run(s, add_noise=False)
        box["ply"] += 1
        if box["ply"] <= temp_moves:
            return int(rng.choice(81, p=pi))
        return int(np.argmax(pi))
    return play


def play_match(p_first, p_second):
    """(선공 결과) +1 선공 승 / 0 무 / -1 후공 승"""
    s, players, turn = start_state(), [p_first, p_second], 0
    for _ in range(200):
        a = players[turn](s)
        s, over, res = apply_move(s, a)
        if over:
            return res if turn == 0 else -res
        turn = 1 - turn
    return 0


# ============================================================ 자기대국 / 학습
def self_play(net, cfg, rng):
    ev = Evaluator(net)
    m = MCTS(ev, cfg["sims"], cfg["c_puct"], cfg["dirichlet_alpha"], cfg["dirichlet_eps"], rng)
    s, trace, ply = start_state(), [], 0
    while True:
        m.reset()
        pi = m.run(s, add_noise=True)
        trace.append((s, pi.copy()))
        a = int(rng.choice(81, p=pi)) if ply < cfg["temp_moves"] else int(np.argmax(pi))
        s, over, res = apply_move(s, a)
        ply += 1
        if over:
            break
        if ply > 160:
            res = 0
            break
    # res는 마지막에 둔 사람 기준. 거기서부터 한 수씩 거슬러 올라가며 부호를 뒤집는다.
    data, z = [], res
    for st, pi in reversed(trace):
        data.append((st, pi, float(z)))
        z = -z
    return data


def train_net(net, buffer, cfg):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs = len(buffer), min(cfg["batch_size"], len(buffer))
    pl = vl = 0.0
    for _ in range(cfg["train_steps"]):
        idx = np.random.choice(n, bs, replace=False)
        x = torch.from_numpy(features([buffer[i][0] for i in idx]))
        tpi = torch.from_numpy(np.stack([buffer[i][1] for i in idx]).astype(np.float32))
        tz = torch.from_numpy(np.array([buffer[i][2] for i in idx], dtype=np.float32))
        logit, v = net(x)
        logp = F.log_softmax(logit.masked_fill(tpi <= 0, -1e9), dim=1)
        lp = -(tpi * logp).sum(1).mean()
        lv = F.mse_loss(v, tz)
        opt.zero_grad()
        (lp + lv).backward()
        opt.step()
        pl += lp.detach().item(); vl += lv.detach().item()
    net.eval()
    return pl / cfg["train_steps"], vl / cfg["train_steps"]


# ============================================================ 여러 프로세스로 나눠 돌리기
_CTX = {}


def _init(sds, cfg):
    torch.set_num_threads(1)
    nets = []
    for sd in sds:
        n = Net(cfg["hidden"], cfg["layers"])
        n.load_state_dict(sd)
        n.eval()
        nets.append(n)
    _CTX.clear(); _CTX.update(nets=nets, cfg=cfg)


def _job_selfplay(seed):
    return self_play(_CTX["nets"][0], _CTX["cfg"], np.random.default_rng(seed))


def _mk(kind, idx, rng):
    cfg = _CTX["cfg"]
    if kind == "random":
        return random_player(rng)
    if kind == "heuristic":
        return heuristic_player(rng)
    return net_player(_CTX["nets"][idx], cfg["eval_sims"], cfg["c_puct"], rng)


def _job_duel(args):
    a_first, ka, ia, kb, ib, seed = args
    rng = np.random.default_rng(seed)
    pa, pb = _mk(ka, ia, rng), _mk(kb, ib, rng)
    r = play_match(pa, pb) if a_first else -play_match(pb, pa)
    return r                                   # A 기준 +1 승 / 0 무 / -1 패


def score(rs):
    return sum(1 if r > 0 else (0.5 if r == 0 else 0) for r in rs) / max(len(rs), 1)


# ============================================================ 성적표
CLR = {"vs_anchor": "#3b82f6", "vs_heuristic": "#f59e0b", "vs_random": "#10b981", "vs_prev": "#ef4444"}
LBL = {"vs_anchor": "기준 신경망 상대", "vs_heuristic": "휴리스틱 상대",
       "vs_random": "무작위 상대", "vs_prev": "이전 버전 상대"}


def svg(series):
    W, H, P = 720, 300, 46
    n = max((len(v) for v in series.values()), default=0)
    if n < 2:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="50"></svg>'
    X = lambda i: P + i * (W - 2 * P) / (n - 1)
    Y = lambda v: H - P - v * (H - 2 * P)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="monospace" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="#14161a"/>']
    for g in (0, .25, .5, .75, 1):
        o.append(f'<line x1="{P}" y1="{Y(g):.1f}" x2="{W-P}" y2="{Y(g):.1f}" stroke="#2a2e35"/>')
        o.append(f'<text x="8" y="{Y(g)+4:.1f}" fill="#7a828e">{int(g*100)}%</text>')
    o.append(f'<line x1="{P}" y1="{Y(.5):.1f}" x2="{W-P}" y2="{Y(.5):.1f}" stroke="#4a5160" stroke-dasharray="4 4"/>')
    for k, vals in series.items():
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals) if v is not None)
        if pts:
            o.append(f'<polyline points="{pts}" fill="none" stroke="{CLR[k]}" stroke-width="2"/>')
    for j, k in enumerate(series):
        cx = P + j * 160
        o.append(f'<rect x="{cx}" y="8" width="9" height="9" fill="{CLR[k]}"/><text x="{cx+14}" y="17" fill="#c7ccd4">{LBL[k]}</text>')
    o.append("</svg>")
    return "\n".join(o)


def write_report(hist):
    os.makedirs(STATE_DIR, exist_ok=True)
    pc = lambda v: "-" if v is None else f"{v*100:.0f}%"
    keys = [k for k in ("vs_anchor", "vs_heuristic", "vs_random", "vs_prev")
            if any(h.get(k) is not None for h in hist)]
    with open(f"{STATE_DIR}/chart.svg", "w") as f:
        f.write(svg({k: [h.get(k) for h in hist] for k in keys}))
    last = hist[-1] if hist else {}
    md = ["# 학습 성적표", "", "![그래프](chart.svg)", "",
          f"- 반복 **{len(hist)}회차**까지 진행",
          f"- 기준 신경망 상대 승률: **{pc(last.get('vs_anchor'))}**  ← 이게 계속 오르면 세지는 중",
          f"- 휴리스틱 상대 승률: **{pc(last.get('vs_heuristic'))}**",
          f"- 무작위 상대 승률: **{pc(last.get('vs_random'))}**",
          f"- 손실: 정책 {last.get('loss_policy')} / 가치 {last.get('loss_value')}",
          f"- 신경망 교체 횟수: {sum(1 for h in hist if h.get('promoted'))}회", "",
          "| 회차 | 기준망 | 휴리스틱 | 무작위 | 이전판 | 교체 | 정책손실 | 가치손실 | 초 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for h in hist[-40:]:
        md.append(f"| {h['iter']} | {pc(h.get('vs_anchor'))} | {pc(h.get('vs_heuristic'))} | "
                  f"{pc(h.get('vs_random'))} | {pc(h.get('vs_prev'))} | {'O' if h.get('promoted') else ''} | "
                  f"{h.get('loss_policy')} | {h.get('loss_value')} | {h.get('sec')} |")
    md += ["", "---", "",
           "**보는 법**", "",
           "- **기준 신경망 상대**가 제일 중요함. 학습 초기에 얼려 둔 신경망과 계속 붙여서, 이 승률이 꾸준히 오르면 진짜로 세지는 중임. 안 오르면 멈춘 것.",
           "- 무작위 상대는 금방 100%에 붙음. 여기서 안 오르면 파이프라인 고장.",
           "- 휴리스틱 상대는 '작은 판 딸 수 있으면 따고, 상대가 딸 자리는 막는' 단순한 봇. 이걸 넘기면 기본기는 된 것.",
           "- 이전 버전 상대 승률이 교체 기준(55%)임. 계속 50% 근처면 더 이상 안 느는 중.",
           "- 정책 손실이 회차가 가도 안 내려가면 탐색이 신경망보다 못하다는 뜻 → `sims`를 올려야 함."]
    with open(f"{STATE_DIR}/REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ============================================================ 본체
def load_progress():
    p = f"{STATE_DIR}/progress.json"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"iter": 0, "history": []}


def save_progress(pr):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = f"{STATE_DIR}/progress.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pr, f, ensure_ascii=False, indent=1)
    os.replace(tmp, f"{STATE_DIR}/progress.json")


def run(cfg_path):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    os.makedirs(STATE_DIR, exist_ok=True)
    workers = cfg.get("workers") or os.cpu_count() or 1
    ctx = mp.get_context("fork")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    deadline = time.time() + cfg["max_minutes"] * 60
    rng = np.random.default_rng()
    pr = load_progress()
    print(f"일꾼 {workers}개 · 목표 {cfg['iterations']}회차 · 현재 {pr['iter']}회차", flush=True)

    best = Net(cfg["hidden"], cfg["layers"])
    if os.path.exists(f"{STATE_DIR}/best.pt"):
        best.load_state_dict(torch.load(f"{STATE_DIR}/best.pt", map_location="cpu"))
        print("이어서 학습", flush=True)
    best.eval()

    anchor = Net(cfg["hidden"], cfg["layers"])
    has_anchor = os.path.exists(f"{STATE_DIR}/anchor.pt")
    if has_anchor:
        anchor.load_state_dict(torch.load(f"{STATE_DIR}/anchor.pt", map_location="cpu"))
    anchor.eval()

    buffer = deque(maxlen=cfg["buffer_size"])
    cpu = lambda m: {k: v.cpu() for k, v in m.state_dict().items()}

    while pr["iter"] < cfg["iterations"]:
        if time.time() > deadline:
            print("시간 예산 종료. 다음 실행에서 이어감.", flush=True)
            break
        t0 = time.time()
        pool = lambda sds: ctx.Pool(workers, initializer=_init, initargs=(sds, cfg))

        with pool([cpu(best)]) as po:
            for d in po.imap_unordered(_job_selfplay, rng.integers(0, 2**31 - 1, cfg["games_per_iter"]).tolist()):
                buffer.extend(d)

        cand = Net(cfg["hidden"], cfg["layers"])
        cand.load_state_dict(best.state_dict())
        lp, lv = train_net(cand, buffer, cfg)

        jobs, na, nt = [], cfg["arena_games"], cfg["test_games"]
        sd = [cpu(cand), cpu(best), cpu(anchor)]
        for i in range(na):
            jobs.append((i % 2 == 0, "net", 0, "net", 1, int(rng.integers(1 << 30))))
        for i in range(nt):
            jobs.append((i % 2 == 0, "net", 1, "random", 0, int(rng.integers(1 << 30))))
        for i in range(nt):
            jobs.append((i % 2 == 0, "net", 1, "heuristic", 0, int(rng.integers(1 << 30))))
        if has_anchor:
            for i in range(nt):
                jobs.append((i % 2 == 0, "net", 1, "net", 2, int(rng.integers(1 << 30))))
        with pool(sd) as po:
            res = po.map(_job_duel, jobs)

        wr = score(res[:na])
        promoted = wr >= cfg["promote_winrate"]
        if promoted:
            best = cand
            torch.save(best.state_dict(), f"{STATE_DIR}/best.pt")
        rec = {"iter": pr["iter"] + 1, "loss_policy": round(lp, 4), "loss_value": round(lv, 4),
               "vs_prev": round(wr, 3), "promoted": promoted,
               "vs_random": round(score(res[na:na + nt]), 3),
               "vs_heuristic": round(score(res[na + nt:na + 2 * nt]), 3),
               "vs_anchor": round(score(res[na + 2 * nt:]), 3) if has_anchor else None,
               "buffer": len(buffer), "sec": round(time.time() - t0, 1)}

        if not has_anchor and pr["iter"] + 1 >= cfg.get("anchor_at", 5):
            torch.save(best.state_dict(), f"{STATE_DIR}/anchor.pt")
            anchor.load_state_dict(best.state_dict()); anchor.eval(); has_anchor = True
            print("   (지금 신경망을 기준점으로 얼려 둠)", flush=True)

        pr["history"].append(rec); pr["iter"] += 1
        save_progress(pr); write_report(pr["history"])
        pc = lambda v: "-" if v is None else f"{v*100:.0f}%"
        print(f"{rec['iter']:>4}회차 손실 p={lp:.3f} v={lv:.3f} | 이전판 {pc(wr)}{' 교체' if promoted else ''}"
              f" | 무작위 {pc(rec['vs_random'])} 휴리스틱 {pc(rec['vs_heuristic'])}"
              f" 기준망 {pc(rec['vs_anchor'])} | {rec['sec']}초", flush=True)

    if pr["iter"] >= cfg["iterations"]:
        with open(f"{STATE_DIR}/DONE", "w") as f:
            f.write("done\n")
        print("모든 회차 완료.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    run(ap.parse_args().config)
