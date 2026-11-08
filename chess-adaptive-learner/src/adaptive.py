"""Motore Adattivo: valuta la forza dell'umano e si calibra per giocare al suo livello.

Valutazione fissa solida (materiale + tavole posizionali PST + mobilita),
ricerca negamax alpha-beta con profondita regolabile. Dopo ogni mossa
dell'umano misura la perdita in centipedi rispetto alla migliore mossa
(riferimento a profondita 3) e a fine partita stima un rating.
"""

import json
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
    PIECE_VALUES,
    QUEEN,
    RANKS,
    ROOK,
    TYPE_MASK,
    WHITE,
)

DEFAULT_STATE_PATH = "adaptive_state.json"
MATE = 100000
MOBILITY_WEIGHT = 2
RATING_WINDOW = 5

# Tavole posizionali classiche (righe da rank 8 a rank 1, prospettiva del bianco).
PAWN_PST = (
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
)
KNIGHT_PST = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)
BISHOP_PST = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)
ROOK_PST = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
)
QUEEN_PST = (
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
)
KING_PST_MG = (
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
)
KING_PST_EG = (
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
)

PST = {PAWN: PAWN_PST, KNIGHT: KNIGHT_PST, BISHOP: BISHOP_PST, ROOK: ROOK_PST, QUEEN: QUEEN_PST}


def _pst_value(table, color, sq):
    """Valore PST per un pezzo su sq: il bianco usa la tavola specchiata in verticale."""
    if color == WHITE:
        return table[(7 - RANKS[sq]) * 8 + FILES[sq]]
    return table[RANKS[sq] * 8 + FILES[sq]]


def _mobility(board, color):
    old = board.turn
    board.turn = color
    n = len(board.generate_pseudo_moves())
    board.turn = old
    return n


class AdaptiveEngine:
    """Motore a valutazione fissa che stima il rating dell'avversario umano."""

    def __init__(self, state_path=DEFAULT_STATE_PATH, seed=None):
        self.state_path = state_path
        self.rng = random.Random(seed)
        self.rating = 1000
        self.games = 0
        self.history = []
        self._losses = []
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.rating = float(data.get("rating", 1000))
            self.games = int(data.get("games", 0))
            self.history = [float(x) for x in data.get("history", [])]

    def evaluate(self, board):
        """Valutazione statica in centipedi dalla prospettiva del bianco."""
        b = board.board
        queens = 0
        score = 0
        for sq in range(64):
            p = b[sq]
            if p == EMPTY:
                continue
            color = p & 8
            t = p & TYPE_MASK
            if t == QUEEN:
                queens += 1
            if t == KING:
                table = KING_PST_EG if queens == 0 else KING_PST_MG
            else:
                table = PST[t]
            val = PIECE_VALUES[t] + _pst_value(table, color, sq)
            score += val if color == WHITE else -val
        score += (_mobility(board, WHITE) - _mobility(board, BLACK)) * MOBILITY_WEIGHT
        return score

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

    def _search_root(self, board, depth):
        """Restituisce la lista [(mossa, punteggio)] ordinata dal migliore, punteggio dalla prospettiva di chi muove."""
        moves = board.legal_moves()
        scored = []
        alpha = -MATE * 2
        for m in moves:
            undo = board.make(m)
            score = -self._negamax(board, depth - 1, -MATE * 2, -alpha)
            board.unmake(m, undo)
            scored.append((m, score))
            if score > alpha:
                alpha = score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def play_depth(self):
        """Profondita di ricerca in funzione del rating stimato."""
        if self.rating < 800:
            return 1
        if self.rating < 1400:
            return 2
        return 3

    def suboptimal_probability(self):
        """Probabilita di scegliere la 2a/3a mossa migliore (decresce col rating)."""
        return max(0.0, min(0.5, 0.5 * (1600.0 - self.rating) / 1600.0))

    def choose_move(self, board):
        """Sceglie la mossa; con probabilita decrescente col rating prende la 2a o 3a migliore."""
        scored = self._search_root(board, self.play_depth())
        if not scored:
            return None
        if len(scored) > 1 and self.rng.random() < self.suboptimal_probability():
            idx = self.rng.choice((1, 2)) if len(scored) > 2 else 1
            return scored[idx][0]
        return scored[0][0]

    def observe_human_move(self, board_before, human_move):
        """Misura la perdita in centipedi della mossa umana rispetto alla migliore (profondita 3)."""
        best = self._search_root(board_before, 3)
        best_score = best[0][1] if best else 0
        undo = board_before.make(human_move)
        human_score = -self._negamax(board_before, 2, -MATE * 2, MATE * 2)
        board_before.unmake(human_move, undo)
        loss = max(0, best_score - human_score)
        self._losses.append(loss)
        return loss

    def on_game_over(self, result, human_color):
        """Aggiorna il rating stimato: media mobile di (2000 - 45 * perdita media), +/-50 sul risultato."""
        avg_loss = sum(self._losses) / len(self._losses) if self._losses else 0.0
        base = 2000.0 - 45.0 * avg_loss
        self.history.append(base)
        window = self.history[-RATING_WINDOW:]
        moving_avg = sum(window) / len(window)
        if result == "1/2-1/2":
            correction = 0.0
        elif (result == "1-0") == (human_color == WHITE):
            correction = 50.0
        else:
            correction = -50.0
        self.rating = max(0.0, min(2000.0, moving_avg + correction))
        self.games += 1
        self._losses = []
        self.save()

    def save(self):
        data = {
            "rating": self.rating,
            "games": self.games,
            "history": self.history[-50:],
        }
        with open(self.state_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def calibration_info(self):
        return {
            "rating": self.rating,
            "games": self.games,
            "profondita": self.play_depth(),
            "prob_subottimale": self.suboptimal_probability(),
        }
