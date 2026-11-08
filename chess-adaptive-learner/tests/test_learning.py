"""Test di apprendimento: convergenza dei pesi e partite simulate complete."""

import json
import os
import random

from src.adaptive import AdaptiveEngine
from src.engine import BLACK, EMPTY, PIECE_VALUES, TYPE_MASK, WHITE, Board
from src.learner import FEATURE_NAMES, LearnerEngine

MATERIAL_KEYS = [
    "materiale_pedoni",
    "materiale_cavalli",
    "materiale_alfieri",
    "materiale_torri",
    "materiale_dama",
]


class MaterialBaseline:
    """Baseline materiale: valuta solo il materiale, ricerca a profondita 1."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def evaluate(self, board):
        score = 0
        for sq in range(64):
            p = board.board[sq]
            if p == EMPTY:
                continue
            val = PIECE_VALUES[p & TYPE_MASK]
            score += val if (p & 8) == WHITE else -val
        return score

    def choose_move(self, board):
        moves = board.legal_moves()
        if not moves:
            return None
        best, best_score = moves[0], -10 ** 9
        for m in moves:
            undo = board.make(m)
            score = -self._negamax(board, 0, -10 ** 9, 10 ** 9)
            board.unmake(m, undo)
            if score > best_score or (score == best_score and self.rng.random() < 0.3):
                best, best_score = m, score
        return best

    def _negamax(self, board, depth, alpha, beta):
        moves = board.legal_moves()
        if not moves:
            return -100000 if board.in_check() else 0
        if depth == 0:
            ev = self.evaluate(board)
            return ev if board.turn == WHITE else -ev
        best = -10 ** 9
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

    def on_game_over(self, result, color):
        pass


def play_game(white_engine, black_engine, max_plies=250, seed=None):
    """Gioca una partita completa tra due motori; restituisce (risultato, motivo)."""
    board = Board()
    engines = {WHITE: white_engine, BLACK: black_engine}
    plies = 0
    result = None
    while plies < max_plies:
        result = board.result()
        if result is not None:
            break
        move = engines[board.turn].choose_move(board)
        if move is None:
            break
        board.make(move)
        plies += 1
    if result is None:
        result = board.result() or ("1/2-1/2", "limite mosse")
    white_engine.on_game_over(result[0], WHITE)
    black_engine.on_game_over(result[0], BLACK)
    return result


def test_convergenza_pesi_dopo_partite_simulate(tmp_path):
    """Dopo N partite learner vs baseline-materiale: pedoni_doppiati < 0 e materiale > iniziale."""
    path = str(tmp_path / "learner_weights.json")
    learner = LearnerEngine(weights_path=path, seed=7, depth=1)
    initial = learner.weights_snapshot()
    baseline = MaterialBaseline(seed=1)
    n_partite = 30
    risultati = []
    for g in range(n_partite):
        if g % 2 == 0:
            risultati.append(play_game(learner, baseline, seed=100 + g))
        else:
            risultati.append(play_game(baseline, learner, seed=100 + g))
    w = learner.weights_snapshot()
    initial_material = sum(initial[k] for k in MATERIAL_KEYS)
    final_material = sum(w[k] for k in MATERIAL_KEYS)
    assert learner.games == n_partite
    assert w["pedoni_doppiati"] < 0, "peso pedoni_doppiati atteso negativo, trovato %s" % w["pedoni_doppiati"]
    assert final_material > initial_material, (
        "somma pesi materiale attesa in aumento: da %s a %s" % (initial_material, final_material)
    )
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["games"] == n_partite
    assert set(data["weights"].keys()) == set(FEATURE_NAMES)


def test_partita_completa_motore_adattivo(tmp_path):
    """Una partita completa del motore adattivo termina senza errori e salva lo stato."""
    state = str(tmp_path / "adaptive_state.json")
    adaptive = AdaptiveEngine(state_path=state, seed=3)
    baseline = MaterialBaseline(seed=5)
    result = play_game(adaptive, baseline, seed=11)
    assert result[0] in ("1-0", "0-1", "1/2-1/2")
    assert os.path.exists(state)
    assert 0 <= adaptive.rating <= 2000
    assert adaptive.games == 1


def test_partita_completa_motore_imparante(tmp_path):
    """Una partita completa del motore imparante (profondita 2) termina senza errori."""
    path = str(tmp_path / "learner_weights.json")
    learner = LearnerEngine(weights_path=path, seed=9, depth=2)
    baseline = MaterialBaseline(seed=2)
    result = play_game(learner, baseline, seed=21)
    assert result[0] in ("1-0", "0-1", "1/2-1/2")
    assert learner.games == 1
    assert os.path.exists(path)


def test_osservazione_mossa_umana_registra_perdita(tmp_path):
    """observe_human_move produce una perdita in centipedi non negativa."""
    adaptive = AdaptiveEngine(state_path=str(tmp_path / "adaptive_state.json"), seed=4)
    board = Board()
    from src.engine import move_from_uci

    mossa = move_from_uci(board, "e2e4")
    loss = adaptive.observe_human_move(board, mossa)
    assert loss >= 0
    # una mossa nettamente peggiore (perde la dama) deve dare perdita maggiore
    b2 = Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    from src.engine import move_from_uci as mf

    brutta = mf(b2, "d1h5")
    loss_brutta = adaptive.observe_human_move(b2, brutta)
    assert loss_brutta >= loss
