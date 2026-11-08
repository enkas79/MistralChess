"""Test del motore: validazione perft e casi speciali delle regole."""

import os

import pytest

from src.engine import (
    DOUBLE,
    QUEEN,
    START_FEN,
    WHITE,
    Board,
    move_from_uci,
    move_to_uci,
    square_from_name,
)


def perft(board, depth):
    return board.perft(depth)


@pytest.mark.parametrize(
    "depth,atteso",
    [(1, 20), (2, 400), (3, 8902), (4, 197281)],
)
def test_perft_posizione_iniziale(depth, atteso):
    """Perft obbligatoria dalla posizione iniziale fino a profondita 4."""
    assert perft(Board(), depth) == atteso


@pytest.mark.skipif(os.environ.get("PERFT_D5") != "1", reason="perft d5 opzionale, abilita con PERFT_D5=1")
def test_perft_posizione_iniziale_d5():
    assert perft(Board(), 5) == 4865609


# Kiwipete: posizione ricca di arrocco, catture e mosse speciali.
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


@pytest.mark.parametrize("depth,atteso", [(1, 48), (2, 2039), (3, 97862)])
def test_perft_kiwipete(depth, atteso):
    assert perft(Board(KIWIPETE), depth) == atteso


# Posizione 3: проверка en passant e spilli (edge cases).
POS3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"


@pytest.mark.parametrize("depth,atteso", [(1, 14), (2, 191), (3, 2812), (4, 43238)])
def test_perft_en_passant_edge_cases(depth, atteso):
    assert perft(Board(POS3), depth) == atteso


# Posizione 4: promozioni e arrocco nero.
POS4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"


@pytest.mark.parametrize("depth,atteso", [(1, 6), (2, 264), (3, 9467)])
def test_perft_promozioni(depth, atteso):
    assert perft(Board(POS4), depth) == atteso


def test_scacco_matto_matto_di_albergo():
    """1.f3 e5 2.g4 Dh4# -> scacco matto per il nero."""
    b = Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        b.make(move_from_uci(b, uci))
    res = b.result()
    assert res == ("0-1", "scacco matto")
    assert b.in_check(WHITE)


def test_stallo():
    """Posizione classica di stallo: nero non ha mosse legali e non e in scacco."""
    b = Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert b.result() == ("1/2-1/2", "stallo")


def test_regola_50_mosse():
    b = Board("k7/8/8/8/8/8/8/1R5K w - - 99 60")
    b.make(move_from_uci(b, "b1b2"))
    assert b.result() == ("1/2-1/2", "regola delle 50 mosse")


def test_materiale_insufficiente():
    assert Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1").result() == ("1/2-1/2", "materiale insufficiente")
    assert Board("8/8/8/4k3/8/8/4B3/4K3 w - - 0 1").result() == ("1/2-1/2", "materiale insufficiente")
    assert Board("8/8/8/4k3/8/8/4N3/4K3 w - - 0 1").result() == ("1/2-1/2", "materiale insufficiente")
    # re + alfiere contro re + alfiere su case dello stesso colore: patta
    b = Board("5b2/8/8/4k3/8/8/8/2B1K3 w - - 0 1")
    assert b.result() == ("1/2-1/2", "materiale insufficiente")
    # re + torre contro re NON e patta automatica
    assert Board("8/8/8/4k3/8/8/8/4K2R w - - 0 1").result() is None


def test_arrocco_non_permesso_attraverso_scacco():
    """Il re non puo arroccare se una casella intermedia e attaccata."""
    # torre nera su f8 attacca f1: l'arrocco corto bianco e illegale
    b = Board("5r2/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    uci = [move_to_uci(m) for m in b.legal_moves()]
    assert "e1g1" not in uci  # e1f1 attaccata dalla torre
    assert "e1c1" in uci      # arrocco lungo ancora legale


def test_arrocco_eseguito_corretamente():
    b = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    b.make(move_from_uci(b, "e1g1"))
    assert b.board[6] == 0x0E  # re bianco su g1 (KING|WHITE)
    assert b.board[5] == 0x0C  # torre su f1 (ROOK|WHITE)
    assert b.board[4] == 0 and b.board[7] == 0
    assert b.castling & 3 == 0  # diritti bianchi revocati


def test_presa_en_passant():
    b = Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        b.make(move_from_uci(b, uci))
    assert b.ep == 43  # d6
    b.make(move_from_uci(b, "e5d6"))
    assert b.board[43] == 0x09  # pedone bianco su d6 (PAWN|WHITE)
    assert b.board[35] == 0     # pedone nero rimosso (d5)


def test_doppia_spinta_imposta_en_passant():
    b = Board()
    m = move_from_uci(b, "e2e4")
    assert m[3] == DOUBLE
    b.make(m)
    assert b.ep == square_from_name("e3")


def test_promozione():
    b = Board("8/P7/8/8/8/8/k6K/8 w - - 0 1")
    m = move_from_uci(b, "a7a8q")
    assert m[2] == QUEEN
    b.make(m)
    assert b.board[56] == QUEEN | WHITE


def test_unmake_ripristina_la_posizione():
    b = Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"):
        prima = b.to_fen()
        m = move_from_uci(b, uci)
        undo = b.make(m)
        b.unmake(m, undo)
        assert b.to_fen() == prima
        b.make(m)
    b2 = Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"):
        b2.make(move_from_uci(b2, uci))
    assert b.to_fen() == b2.to_fen()


def test_fen_roundtrip():
    for fen in (START_FEN, KIWIPETE, POS3, POS4):
        assert Board(fen).to_fen() == fen


def test_partita_completa_senza_errori():
    """Una partita casuale (prima mossa legale) termina o arriva al limite mosse."""
    import random

    rng = random.Random(42)
    b = Board()
    for _ in range(300):
        if b.result() is not None:
            break
        moves = b.legal_moves()
        b.make(rng.choice(moves))
    assert b.result() is not None or b.fullmove > 1
