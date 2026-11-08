"""Motore Imparante: sa solo le regole all'inizio e migliora partita dopo partita.

Valutazione = combinazione lineare di 16 feature con pesi iniziali casuali in
[-0.5, 0.5]. Ricerca negamax con alpha-beta (profondita 2-3). A fine partita
aggiorna i pesi con regressione logistica verso l'esito e salva su disco.
"""

import json
import math
import os
import random

from .engine import (
    BISHOP,
    BLACK,
    EMPTY,
    FILES,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    RANKS,
    ROOK,
    TYPE_MASK,
    WHITE,
    Board,
)

FEATURE_NAMES = [
    "materiale_pedoni",
    "materiale_cavalli",
    "materiale_alfieri",
    "materiale_torri",
    "materiale_dama",
    "mobilita",
    "coppia_alfieri",
    "pedoni_doppiati",
    "pedoni_isolati",
    "pedoni_passati",
    "scudo_pedoni_re",
    "controllo_centro",
    "torri_colonna_aperta",
    "pressione_re_avversario",
    "sviluppo",
    "avanzamento_pedoni",
]

DEFAULT_WEIGHTS_PATH = "learner_weights.json"
LEARNING_RATE = 0.02
MATE = 100000

CENTER_SQUARES = (27, 28, 35, 36)  # d4, e4, d5, e5


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _mobility(board, color):
    """Numero di mosse pseudo-legali del colore indicato (senza toccare lo stato)."""
    old = board.turn
    board.turn = color
    n = len(board.generate_pseudo_moves())
    board.turn = old
    return n


def extract_features(board):
    """Vettore di 16 feature dalla prospettiva del bianco (positivo = meglio per il bianco)."""
    b = board.board
    counts = {WHITE: {PAWN: 0, KNIGHT: 0, BISHOP: 0, ROOK: 0, QUEEN: 0}, BLACK: {PAWN: 0, KNIGHT: 0, BISHOP: 0, ROOK: 0, QUEEN: 0}}
    pawns = {WHITE: [], BLACK: []}
    rooks = {WHITE: [], BLACK: []}
    for sq in range(64):
        p = b[sq]
        if p == EMPTY:
            continue
        color = p & 8
        t = p & TYPE_MASK
        if t == KING:
            continue
        counts[color][t] += 1
        if t == PAWN:
            pawns[color].append(sq)
        elif t == ROOK:
            rooks[color].append(sq)

    f = {}
    for t, name in ((PAWN, "materiale_pedoni"), (KNIGHT, "materiale_cavalli"), (BISHOP, "materiale_alfieri"),
                    (ROOK, "materiale_torri"), (QUEEN, "materiale_dama")):
        f[name] = counts[WHITE][t] - counts[BLACK][t]

    f["mobilita"] = (_mobility(board, WHITE) - _mobility(board, BLACK)) * 0.1
    f["coppia_alfieri"] = (1 if counts[WHITE][BISHOP] >= 2 else 0) - (1 if counts[BLACK][BISHOP] >= 2 else 0)

    def pawn_stats(color):
        per_file = {}
        for sq in pawns[color]:
            per_file.setdefault(FILES[sq], []).append(RANKS[sq])
        doubled = sum(len(v) - 1 for v in per_file.values() if len(v) > 1)
        isolated = 0
        passed = 0
        enemy = pawns[BLACK if color == WHITE else WHITE]
        enemy_files = {}
        for sq in enemy:
            enemy_files.setdefault(FILES[sq], []).append(RANKS[sq])
        for sq in pawns[color]:
            fl, rk = FILES[sq], RANKS[sq]
            if fl - 1 not in per_file and fl + 1 not in per_file:
                isolated += 1
            dir_sign = 1 if color == WHITE else -1
            blocked = False
            for ef in (fl - 1, fl, fl + 1):
                for er in enemy_files.get(ef, ()):
                    if (er - rk) * dir_sign > 0:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                passed += 1
        return doubled, isolated, passed

    d_w, i_w, p_w = pawn_stats(WHITE)
    d_b, i_b, p_b = pawn_stats(BLACK)
    f["pedoni_doppiati"] = d_w - d_b
    f["pedoni_isolati"] = i_w - i_b
    f["pedoni_passati"] = p_w - p_b

    def king_shield(color):
        ksq = board.king_sq[0 if color == WHITE else 1]
        kf, kr = FILES[ksq], RANKS[ksq]
        dir_sign = 1 if color == WHITE else -1
        n = 0
        for df in (-1, 0, 1):
            nf = kf + df
            if 0 <= nf < 8:
                for dr in (1, 2):
                    nr = kr + dr * dir_sign
                    if 0 <= nr < 8 and b[nr * 8 + nf] == (PAWN | color):
                        n += 1
        return n

    f["scudo_pedoni_re"] = king_shield(WHITE) - king_shield(BLACK)

    cw = sum(1 for s in CENTER_SQUARES if board.is_attacked(s, WHITE))
    cb = sum(1 for s in CENTER_SQUARES if board.is_attacked(s, BLACK))
    f["controllo_centro"] = cw - cb

    pawn_files = set(FILES[sq] for sq in pawns[WHITE] + pawns[BLACK])
    open_rooks = 0
    for color in (WHITE, BLACK):
        n = sum(1 for sq in rooks[color] if FILES[sq] not in pawn_files)
        open_rooks += n if color == WHITE else -n
    f["torri_colonna_aperta"] = open_rooks

    def pressure(color):
        enemy_king = board.king_sq[1 if color == WHITE else 0]
        kf, kr = FILES[enemy_king], RANKS[enemy_king]
        n = 0
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                nf, nr = kf + df, kr + dr
                if 0 <= nf < 8 and 0 <= nr < 8 and board.is_attacked(nr * 8 + nf, color):
                    n += 1
        return n

    f["pressione_re_avversario"] = pressure(WHITE) - pressure(BLACK)

    dev_w = sum(1 for sq in range(64) if b[sq] in (KNIGHT | WHITE, BISHOP | WHITE) and RANKS[sq] > 0)
    dev_b = sum(1 for sq in range(64) if b[sq] in (KNIGHT, BISHOP) and RANKS[sq] < 7)
    f["sviluppo"] = dev_w - dev_b

    adv = sum(RANKS[sq] for sq in pawns[WHITE]) - sum(7 - RANKS[sq] for sq in pawns[BLACK])
    f["avanzamento_pedoni"] = adv * 0.1

    return f


class LearnerEngine:
    """Motore con pesi appresi via regressione logistica sull'esito delle partite."""

    def __init__(self, weights_path=DEFAULT_WEIGHTS_PATH, seed=None, depth=2):
        self.weights_path = weights_path
        self.depth = depth
        self.rng = random.Random(seed)
        self.games = 0
        self.weights = {}
        if os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.weights = {k: float(v) for k, v in data["weights"].items()}
            self.games = int(data.get("games", 0))
            for name in FEATURE_NAMES:
                self.weights.setdefault(name, self.rng.uniform(-0.5, 0.5))
        else:
            self.weights = {name: self.rng.uniform(-0.5, 0.5) for name in FEATURE_NAMES}
        self.color = WHITE
        self._game_features = []

    def evaluate(self, board):
        """Valutazione statica dalla prospettiva del bianco."""
        f = extract_features(board)
        return sum(self.weights[k] * f[k] for k in FEATURE_NAMES)

    def _negamax(self, board, depth, alpha, beta):
        if board.halfmove >= 100:
            return 0
        moves = board.legal_moves()
        if not moves:
            if board.in_check():
                return -(MATE + depth)
            return 0
        if depth == 0:
            if board.insufficient_material():
                return 0
            ev = self.evaluate(board)
            return ev if board.turn == WHITE else -ev
        if self.rng.random() < 0.3:
            self.rng.shuffle(moves)
        best = -MATE * 2
        for m in moves:
            undo = board.make(m)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.unmake(m, undo)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def choose_move(self, board):
        """Sceglie la mossa (registra la posizione per l'apprendimento a fine partita)."""
        moves = board.legal_moves()
        if not moves:
            return None
        self._game_features.append(extract_features(board))
        if self.rng.random() < 0.5:
            self.rng.shuffle(moves)
        best_move = moves[0]
        best_score = -MATE * 2
        alpha = -MATE * 2
        for m in moves:
            undo = board.make(m)
            score = -self._negamax(board, self.depth - 1, -MATE * 2, -alpha)
            board.unmake(m, undo)
            if score > best_score:
                best_score = score
                best_move = m
            if score > alpha:
                alpha = score
        return best_move

    def on_game_over(self, result, color=None):
        """Aggiorna i pesi con regressione logistica verso l'esito e salva su disco.

        result: "1-0" | "0-1" | "1/2-1/2"; color: colore del motore.
        """
        if color is None:
            color = self.color
        if result == "1/2-1/2":
            outcome = 0.5
        elif (result == "1-0") == (color == WHITE):
            outcome = 1.0
        else:
            outcome = 0.0
        for f in self._game_features:
            z = sum(self.weights[k] * f[k] for k in FEATURE_NAMES)
            s = sigmoid(z)
            for k in FEATURE_NAMES:
                self.weights[k] += LEARNING_RATE * (outcome - s) * f[k]
        self._game_features = []
        self.games += 1
        self.save()

    def save(self):
        data = {"weights": self.weights, "games": self.games}
        with open(self.weights_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def weights_snapshot(self):
        return dict(self.weights)
