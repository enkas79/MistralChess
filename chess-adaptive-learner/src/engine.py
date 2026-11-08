"""Motore scacchistico autoimplementato in Python puro.

Rappresentazione mailbox 64 (a1 = 0, h8 = 63) con make/unmake.
Nessuna libreria esterna: solo regole degli scacchi.

Pezzo = tipo (1..6) | bit colore (8 = bianco, 0 = nero). Casella vuota = 0.
Mossa = tupla (from_sq, to_sq, promozione, flag) con flag in
NORMAL, DOUBLE (doppia spinta), EP (presa en passant), CASTLE (arrocco).
"""

EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
WHITE, BLACK = 8, 0
COLOR_BIT = 8
TYPE_MASK = 7

WP, WN, WB, WR, WQ, WK = PAWN | WHITE, KNIGHT | WHITE, BISHOP | WHITE, ROOK | WHITE, QUEEN | WHITE, KING | WHITE
BP, BN, BB, BR, BQ, BK = PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

PIECE_VALUES = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 0}

NORMAL, DOUBLE, EP, CASTLE = 0, 1, 2, 3

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

FILES = [sq % 8 for sq in range(64)]
RANKS = [sq // 8 for sq in range(64)]

KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_DELTAS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8


def _ci(color):
    """Indice colore per le tabelle: 0 = bianco, 1 = nero."""
    return 0 if color == WHITE else 1


def _build_tables():
    knight_atk = [[] for _ in range(64)]
    king_atk = [[] for _ in range(64)]
    pawn_atk = [[[] for _ in range(64)] for _ in range(2)]
    pawn_attackers = [[[] for _ in range(64)] for _ in range(2)]
    for sq in range(64):
        f, r = FILES[sq], RANKS[sq]
        for df, dr in KNIGHT_DELTAS:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                knight_atk[sq].append(nr * 8 + nf)
        for df, dr in KING_DELTAS:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                king_atk[sq].append(nr * 8 + nf)
        for df in (-1, 1):
            nf = f + df
            nr = r + 1
            if 0 <= nf < 8 and nr < 8:
                pawn_atk[0][sq].append(nr * 8 + nf)
            nr = r - 1
            if 0 <= nf < 8 and nr >= 0:
                pawn_atk[1][sq].append(nr * 8 + nf)
    for ci in (0, 1):
        for s in range(64):
            for t in pawn_atk[ci][s]:
                pawn_attackers[ci][t].append(s)
    return knight_atk, king_atk, pawn_atk, pawn_attackers


KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS, PAWN_ATTACKERS = _build_tables()

PIECE_SYMBOLS = {
    WP: "P", WN: "N", WB: "B", WR: "R", WQ: "Q", WK: "K",
    BP: "p", BN: "n", BB: "b", BR: "r", BQ: "q", BK: "k",
}
SYMBOL_TO_PIECE = {v: k for k, v in PIECE_SYMBOLS.items()}


def square_name(sq):
    return chr(ord("a") + FILES[sq]) + str(RANKS[sq] + 1)


def square_from_name(name):
    return (int(name[1]) - 1) * 8 + (ord(name[0]) - ord("a"))


def move_to_uci(move):
    frm, to, promo, _flag = move
    s = square_name(frm) + square_name(to)
    if promo:
        s += {QUEEN: "q", ROOK: "r", BISHOP: "b", KNIGHT: "n"}[promo]
    return s


def move_from_uci(board, uci):
    """Converte una stringa UCI in mossa legale della posizione (promozione Inclusa)."""
    frm = square_from_name(uci[0:2])
    to = square_from_name(uci[2:4])
    promo = 0
    if len(uci) == 5:
        promo = {"q": QUEEN, "r": ROOK, "b": BISHOP, "n": KNIGHT}[uci[4]]
    for m in board.legal_moves():
        if m[0] == frm and m[1] == to and (promo == 0 or m[2] == promo):
            return m
    raise ValueError("mossa illegale: %s" % uci)


class Board:
    """Scacchiera con make/unmake e generazione mosse legali completa."""

    def __init__(self, fen=START_FEN):
        self.board = [EMPTY] * 64
        self.king_sq = [0, 0]
        self._load_fen(fen)

    def _load_fen(self, fen):
        parts = fen.split()
        rows = parts[0].split("/")
        for r, row in enumerate(rows):
            rank = 7 - r
            file = 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                else:
                    piece = SYMBOL_TO_PIECE[ch]
                    sq = rank * 8 + file
                    self.board[sq] = piece
                    if piece & TYPE_MASK == KING:
                        self.king_sq[_ci(piece & COLOR_BIT)] = sq
                    file += 1
        self.turn = WHITE if parts[1] == "w" else BLACK
        castling = 0
        if len(parts) > 2 and parts[2] != "-":
            if "K" in parts[2]:
                castling |= CASTLE_WK
            if "Q" in parts[2]:
                castling |= CASTLE_WQ
            if "k" in parts[2]:
                castling |= CASTLE_BK
            if "q" in parts[2]:
                castling |= CASTLE_BQ
        self.castling = castling
        self.ep = -1 if len(parts) <= 3 or parts[3] == "-" else square_from_name(parts[3])
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1

    def to_fen(self):
        rows = []
        for rank in range(7, -1, -1):
            row = ""
            empty = 0
            for file in range(8):
                p = self.board[rank * 8 + file]
                if p == EMPTY:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += PIECE_SYMBOLS[p]
            if empty:
                row += str(empty)
            rows.append(row)
        cr = ""
        if self.castling & CASTLE_WK:
            cr += "K"
        if self.castling & CASTLE_WQ:
            cr += "Q"
        if self.castling & CASTLE_BK:
            cr += "k"
        if self.castling & CASTLE_BQ:
            cr += "q"
        if not cr:
            cr = "-"
        ep = "-" if self.ep < 0 else square_name(self.ep)
        return "%s %s %s %s %d %d" % ("/".join(rows), "w" if self.turn == WHITE else "b", cr, ep, self.halfmove, self.fullmove)

    def copy(self):
        b = Board.__new__(Board)
        b.board = self.board[:]
        b.king_sq = self.king_sq[:]
        b.turn = self.turn
        b.castling = self.castling
        b.ep = self.ep
        b.halfmove = self.halfmove
        b.fullmove = self.fullmove
        return b

    def is_attacked(self, sq, by):
        """True se la casella sq e attaccata dal colore `by` (WHITE/BLACK)."""
        board = self.board
        ci = _ci(by)
        pawn = PAWN | by
        for s in PAWN_ATTACKERS[ci][sq]:
            if board[s] == pawn:
                return True
        knight = KNIGHT | by
        for s in KNIGHT_ATTACKS[sq]:
            if board[s] == knight:
                return True
        king = KING | by
        for s in KING_ATTACKS[sq]:
            if board[s] == king:
                return True
        bishop_q = BISHOP | by
        queen_b = QUEEN | by
        f, r = FILES[sq], RANKS[sq]
        for df, dr in BISHOP_DIRS:
            nf, nr = f + df, r + dr
            while 0 <= nf < 8 and 0 <= nr < 8:
                p = board[nr * 8 + nf]
                if p:
                    if p == bishop_q or p == queen_b:
                        return True
                    break
                nf += df
                nr += dr
        rook_q = ROOK | by
        for df, dr in ROOK_DIRS:
            nf, nr = f + df, r + dr
            while 0 <= nf < 8 and 0 <= nr < 8:
                p = board[nr * 8 + nf]
                if p:
                    if p == rook_q or p == queen_b:
                        return True
                    break
                nf += df
                nr += dr
        return False

    def in_check(self, color=None):
        if color is None:
            color = self.turn
        return self.is_attacked(self.king_sq[_ci(color)], color ^ COLOR_BIT)

    def generate_pseudo_moves(self):
        """Mosse pseudo-legali (il re potrebbe restare sotto scacco)."""
        moves = []
        board = self.board
        turn = self.turn
        enemy = turn ^ COLOR_BIT
        ci = _ci(turn)
        ep = self.ep
        for frm in range(64):
            p = board[frm]
            if p == EMPTY or (p & COLOR_BIT) != turn:
                continue
            t = p & TYPE_MASK
            if t == PAWN:
                fwd = 8 if turn == WHITE else -8
                to = frm + fwd
                promo_rank = 7 if turn == WHITE else 0
                if board[to] == EMPTY:
                    if RANKS[to] == promo_rank:
                        for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                            moves.append((frm, to, promo, NORMAL))
                    else:
                        moves.append((frm, to, 0, NORMAL))
                        start_rank = 1 if turn == WHITE else 6
                        if RANKS[frm] == start_rank and board[frm + 2 * fwd] == EMPTY:
                            moves.append((frm, frm + 2 * fwd, 0, DOUBLE))
                for to in PAWN_ATTACKS[ci][frm]:
                    q = board[to]
                    if q != EMPTY and (q & COLOR_BIT) == enemy:
                        if RANKS[to] == promo_rank:
                            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                                moves.append((frm, to, promo, NORMAL))
                        else:
                            moves.append((frm, to, 0, NORMAL))
                    elif to == ep:
                        moves.append((frm, to, 0, EP))
            elif t == KNIGHT or t == KING:
                table = KNIGHT_ATTACKS if t == KNIGHT else KING_ATTACKS
                for to in table[frm]:
                    q = board[to]
                    if q == EMPTY or (q & COLOR_BIT) == enemy:
                        moves.append((frm, to, 0, NORMAL))
                if t == KING:
                    if turn == WHITE and frm == 4:
                        if (self.castling & CASTLE_WK and board[5] == EMPTY and board[6] == EMPTY
                                and not self.is_attacked(4, BLACK) and not self.is_attacked(5, BLACK)
                                and not self.is_attacked(6, BLACK)):
                            moves.append((4, 6, 0, CASTLE))
                        if (self.castling & CASTLE_WQ and board[3] == EMPTY and board[2] == EMPTY
                                and board[1] == EMPTY and not self.is_attacked(4, BLACK)
                                and not self.is_attacked(3, BLACK) and not self.is_attacked(2, BLACK)):
                            moves.append((4, 2, 0, CASTLE))
                    elif turn == BLACK and frm == 60:
                        if (self.castling & CASTLE_BK and board[61] == EMPTY and board[62] == EMPTY
                                and not self.is_attacked(60, WHITE) and not self.is_attacked(61, WHITE)
                                and not self.is_attacked(62, WHITE)):
                            moves.append((60, 62, 0, CASTLE))
                        if (self.castling & CASTLE_BQ and board[59] == EMPTY and board[58] == EMPTY
                                and board[57] == EMPTY and not self.is_attacked(60, WHITE)
                                and not self.is_attacked(59, WHITE) and not self.is_attacked(58, WHITE)):
                            moves.append((60, 58, 0, CASTLE))
            else:
                if t == BISHOP:
                    dirs = BISHOP_DIRS
                elif t == ROOK:
                    dirs = ROOK_DIRS
                else:
                    dirs = BISHOP_DIRS + ROOK_DIRS
                f, r = FILES[frm], RANKS[frm]
                for df, dr in dirs:
                    nf, nr = f + df, r + dr
                    while 0 <= nf < 8 and 0 <= nr < 8:
                        to = nr * 8 + nf
                        q = board[to]
                        if q == EMPTY:
                            moves.append((frm, to, 0, NORMAL))
                        else:
                            if (q & COLOR_BIT) == enemy:
                                moves.append((frm, to, 0, NORMAL))
                            break
                        nf += df
                        nr += dr
        return moves

    def legal_moves(self):
        """Mosse legali: pseudo-legali filtrate con make/unmake + re sotto scacco."""
        res = []
        for m in self.generate_pseudo_moves():
            undo = self.make(m)
            mover_ci = 0 if self.turn == BLACK else 1
            if not self.is_attacked(self.king_sq[mover_ci], self.turn):
                res.append(m)
            self.unmake(m, undo)
        return res

    def make(self, move):
        """Esegue la mossa e restituisce il token per unmake."""
        frm, to, promo, flag = move
        board = self.board
        piece = board[frm]
        captured = board[to]
        cap_sq = to
        if flag == EP:
            cap_sq = to - 8 if self.turn == WHITE else to + 8
            captured = board[cap_sq]
        undo = (piece, captured, cap_sq, self.castling, self.ep, self.halfmove, self.fullmove)
        if flag == CASTLE:
            if to == 6:
                board[5], board[7] = board[7], EMPTY
            elif to == 2:
                board[3], board[0] = board[0], EMPTY
            elif to == 62:
                board[61], board[63] = board[63], EMPTY
            elif to == 58:
                board[59], board[56] = board[56], EMPTY
        board[to] = (promo | self.turn) if promo else piece
        board[frm] = EMPTY
        if flag == EP:
            board[cap_sq] = EMPTY
        if (piece & TYPE_MASK) == KING:
            self.king_sq[_ci(self.turn)] = to
        cr = self.castling
        if piece == WK:
            cr &= ~(CASTLE_WK | CASTLE_WQ)
        elif piece == BK:
            cr &= ~(CASTLE_BK | CASTLE_BQ)
        if frm == 0 or to == 0:
            cr &= ~CASTLE_WQ
        if frm == 7 or to == 7:
            cr &= ~CASTLE_WK
        if frm == 56 or to == 56:
            cr &= ~CASTLE_BQ
        if frm == 63 or to == 63:
            cr &= ~CASTLE_BK
        self.castling = cr
        self.ep = (frm + to) // 2 if flag == DOUBLE else -1
        if (piece & TYPE_MASK) == PAWN or captured != EMPTY:
            self.halfmove = 0
        else:
            self.halfmove += 1
        if self.turn == BLACK:
            self.fullmove += 1
        self.turn ^= COLOR_BIT
        return undo

    def unmake(self, move, undo):
        """Annulla la mossa usando il token restituito da make."""
        frm, to, promo, flag = move
        piece, captured, cap_sq, castling, ep, halfmove, fullmove = undo
        self.turn ^= COLOR_BIT
        board = self.board
        board[frm] = piece
        board[to] = EMPTY
        if cap_sq >= 0 and cap_sq != to:
            board[cap_sq] = captured
        elif cap_sq == to:
            board[to] = captured
        if flag == CASTLE:
            if to == 6:
                board[7], board[5] = board[5], EMPTY
            elif to == 2:
                board[0], board[3] = board[3], EMPTY
            elif to == 62:
                board[63], board[61] = board[61], EMPTY
            elif to == 58:
                board[56], board[59] = board[59], EMPTY
        if (piece & TYPE_MASK) == KING:
            self.king_sq[_ci(self.turn)] = frm
        self.castling = castling
        self.ep = ep
        self.halfmove = halfmove
        self.fullmove = fullmove

    def perft(self, depth):
        """Conteggio nodi dell'albero delle mosse legali fino a `depth`."""
        if depth == 0:
            return 1
        moves = self.legal_moves()
        if depth == 1:
            return len(moves)
        n = 0
        for m in moves:
            undo = self.make(m)
            n += self.perft(depth - 1)
            self.unmake(m, undo)
        return n

    def insufficient_material(self):
        """Vero se nessun giocatore puo dare matto (K vs K, K+minore vs K, K+B vs K+B stesso colore)."""
        pieces = []
        for sq in range(64):
            p = self.board[sq]
            if p and (p & TYPE_MASK) != KING:
                pieces.append((p & TYPE_MASK, p & COLOR_BIT, sq))
        if not pieces:
            return True
        if len(pieces) == 1 and pieces[0][0] in (BISHOP, KNIGHT):
            return True
        if len(pieces) == 2 and all(t == BISHOP for t, _c, _s in pieces):
            c0 = pieces[0][1]
            c1 = pieces[1][1]
            if c0 != c1:
                sq0, sq1 = pieces[0][2], pieces[1][2]
                if (FILES[sq0] + RANKS[sq0]) % 2 == (FILES[sq1] + RANKS[sq1]) % 2:
                    return True
        return False

    def result(self):
        """None se la partita continua, altrimenti (risultato, motivo).

        risultato in notazione standard: "1-0", "0-1", "1/2-1/2".
        """
        if self.insufficient_material():
            return ("1/2-1/2", "materiale insufficiente")
        if self.halfmove >= 100:
            return ("1/2-1/2", "regola delle 50 mosse")
        moves = self.legal_moves()
        if not moves:
            if self.in_check():
                return ("0-1" if self.turn == WHITE else "1-0", "scacco matto")
            return ("1/2-1/2", "stallo")
        return None
