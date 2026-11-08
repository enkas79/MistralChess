"""Interfaccia grafica PySide6 (Qt 6).

Scacchiera in QGraphicsView/QGraphicsScene con caselle cliccabili, evidenziazione
della selezione e delle mosse legali, coordinate e pezzi Unicode. Il calcolo
della mossa del motore avviene in un QThread dedicato che emette un signal:
mai chiamate dirette ai widget dal thread di lavoro.
"""

from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .adaptive import AdaptiveEngine
from .engine import (
    BISHOP,
    BLACK,
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
from .learner import FEATURE_NAMES, LearnerEngine

SQ = 64          # dimensione di una casella in pixel
MARGIN = 28      # bordo per le coordinate
BOARD_PX = SQ * 8

# Glifi Unicode dei pezzi (bianco vuoto / nero pieno).
GLYPH_WHITE = {KING: "♔", QUEEN: "♕", ROOK: "♖", BISHOP: "♗", KNIGHT: "♘", PAWN: "♙"}
GLYPH_BLACK = {KING: "♚", QUEEN: "♛", ROOK: "♜", BISHOP: "♝", KNIGHT: "♞", PAWN: "♟"}

COLOR_LIGHT = QColor("#f0d9b5")
COLOR_DARK = QColor("#b58863")
COLOR_SELECTED = QColor(255, 235, 59, 110)
COLOR_LEGAL = QColor(20, 120, 20, 90)
COLOR_LASTMOVE = QColor(80, 160, 255, 70)


class EngineWorker(QThread):
    """Thread di lavoro: calcola la mossa del motore fuori dal main thread."""

    move_ready = Signal(object)

    def __init__(self, engine, board_fen, board_fen_before=None, human_move=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.board_fen = board_fen
        self.board_fen_before = board_fen_before
        self.human_move = human_move

    def run(self):
        board = Board(self.board_fen)
        if self.board_fen_before is not None and isinstance(self.engine, AdaptiveEngine):
            prima = Board(self.board_fen_before)
            self.engine.observe_human_move(prima, self.human_move)
        move = self.engine.choose_move(board)
        self.move_ready.emit(move)


class ChessView(QGraphicsView):
    """Vista con gestione del click sulle caselle."""

    def __init__(self, on_square, parent=None):
        super().__init__(parent)
        self._on_square = on_square
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setFixedSize(BOARD_PX + 2 * MARGIN, BOARD_PX + 2 * MARGIN)

    def scene_rect(self):
        return self._scene.sceneRect()

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        col = int((pos.x() - MARGIN) // SQ)
        row = int((pos.y() - MARGIN) // SQ)
        if 0 <= col < 8 and 0 <= row < 8:
            self._on_square(col, row)
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scacchi Adattivo/Imparante")
        self.board = Board()
        self.history = [self.board.to_fen()]
        self.mode = "adaptive"
        self.player_color = WHITE
        self.engine = None
        self.worker = None
        self.selected = None       # casella (file, rank) selezionata, in coordinate display
        self.legal_cache = []
        self.last_move = None      # (from_sq, to_sq)
        self._game_over = False
        self._ready = False
        self._build_ui()
        self._ready = True
        self.new_game()

    # ---------- costruzione UI ----------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.view = ChessView(self.on_display_square)
        root.addWidget(self.view)

        panel = QVBoxLayout()
        root.addLayout(panel, 1)

        grp_mode = QGroupBox("Modalità")
        lay_mode = QVBoxLayout(grp_mode)
        self.radio_adaptive = QRadioButton("Motore Adattivo (si calibra sul tuo livello)")
        self.radio_learner = QRadioButton("Motore Imparante (impara partita dopo partita)")
        self.radio_adaptive.setChecked(True)
        self.radio_adaptive.toggled.connect(self.on_mode_changed)
        self.radio_learner.toggled.connect(self.on_mode_changed)
        lay_mode.addWidget(self.radio_adaptive)
        lay_mode.addWidget(self.radio_learner)
        panel.addWidget(grp_mode)

        grp_color = QGroupBox("Il tuo colore")
        lay_color = QVBoxLayout(grp_color)
        self.radio_white = QRadioButton("Bianco")
        self.radio_black = QRadioButton("Nero")
        self.radio_white.setChecked(True)
        lay_color.addWidget(self.radio_white)
        lay_color.addWidget(self.radio_black)
        panel.addWidget(grp_color)

        self.btn_new = QPushButton("Nuova partita")
        self.btn_new.clicked.connect(self.new_game)
        self.btn_undo = QPushButton("Annulla mossa")
        self.btn_undo.clicked.connect(self.undo_move)
        panel.addWidget(self.btn_new)
        panel.addWidget(self.btn_undo)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        panel.addWidget(self.status)

        self.stack = QStackedWidget()
        panel.addWidget(self.stack, 1)

        # Pagina adattivo: rating stimato e livello di calibrazione.
        page_ad = QWidget()
        lay_ad = QVBoxLayout(page_ad)
        self.lbl_rating = QLabel("")
        self.lbl_calibrazione = QLabel("")
        lay_ad.addWidget(self.lbl_rating)
        lay_ad.addWidget(self.lbl_calibrazione)
        self.stack.addWidget(page_ad)

        # Pagina imparante: barre +/- dei pesi appresi.
        page_lr = QWidget()
        lay_lr = QGridLayout(page_lr)
        self.lbl_games = QLabel("")
        lay_lr.addWidget(self.lbl_games, 0, 0, 1, 2)
        self.weight_bars = {}
        for i, name in enumerate(FEATURE_NAMES, start=1):
            lay_lr.addWidget(QLabel(name), i, 0)
            bar = QProgressBar()
            bar.setRange(-100, 100)
            bar.setValue(0)
            bar.setFormat("%.2f")
            self.weight_bars[name] = bar
            lay_lr.addWidget(bar, i, 1)
        self.stack.addWidget(page_lr)

    # ---------- logica di gioco ----------

    def new_game(self):
        self._stop_worker()
        self.board = Board()
        self.history = [self.board.to_fen()]
        self.selected = None
        self.legal_cache = []
        self.last_move = None
        self._game_over = False
        if self.radio_adaptive.isChecked():
            self.mode = "adaptive"
            self.engine = AdaptiveEngine()
        else:
            self.mode = "learner"
            self.engine = LearnerEngine()
        self.player_color = WHITE if self.radio_white.isChecked() else BLACK
        self.engine.color = BLACK if self.player_color == WHITE else WHITE
        self.redraw()
        self.update_panel()
        self.status.setText("Nuova partita: muovi tu." if self.player_color == self.board.turn
                            else "Nuova partita: turno del motore...")
        if self.board.turn != self.player_color:
            self._start_engine_move()

    def on_mode_changed(self):
        if not self._ready:
            return
        sender = self.sender()
        if sender is not None and not sender.isChecked():
            return
        self.new_game()

    def on_display_square(self, col, row):
        """Click su una casella in coordinate display (colonna 0 = sinistra)."""
        sq = self.display_to_square(col, row)
        if sq is None or self._game_over or self._engine_thinking():
            return
        if self.board.turn != self.player_color:
            return
        piece = self.board.board[sq]
        if self.selected is not None:
            for m in self.legal_cache:
                if m[1] == sq:
                    self._make_human_move(m)
                    return
        if piece != 0 and (piece & 8) == self.player_color:
            self.selected = (col, row)
            self.legal_cache = [m for m in self.board.legal_moves() if m[0] == sq]
        else:
            self.selected = None
            self.legal_cache = []
        self.redraw()

    def _make_human_move(self, move):
        """Applica la mossa dell'umano; promozione sempre a dama."""
        if move[2] != 0:
            for m in self.legal_cache:
                if m[0] == move[0] and m[1] == move[1] and m[2] == QUEEN:
                    move = m
                    break
        fen_before = self.board.to_fen()
        self.last_move = (move[0], move[1])
        self.board.make(move)
        self.history.append(self.board.to_fen())
        self.selected = None
        self.legal_cache = []
        self.redraw()
        if self._check_game_over():
            return
        self.status.setText("Turno del motore...")
        self._start_engine_move(fen_before, move)

    def _start_engine_move(self, fen_before=None, human_move=None):
        self._stop_worker()
        self.worker = EngineWorker(
            self.engine,
            self.board.to_fen(),
            board_fen_before=fen_before,
            human_move=human_move,
            parent=self,
        )
        self.worker.move_ready.connect(self._on_engine_move)
        self.worker.start()

    def _on_engine_move(self, move):
        if self.sender() is not self.worker or self._game_over:
            return
        if move is None:
            self._check_game_over()
            return
        self.last_move = (move[0], move[1])
        self.board.make(move)
        self.history.append(self.board.to_fen())
        self.redraw()
        if self._check_game_over():
            return
        self.status.setText("Tocca a te.")
        self.update_panel()

    def _check_game_over(self):
        res = self.board.result()
        if res is None:
            return False
        self._game_over = True
        self.status.setText("Partita terminata: %s (%s)" % (res[0], res[1]))
        if self.engine is not None:
            if isinstance(self.engine, AdaptiveEngine):
                self.engine.on_game_over(res[0], self.player_color)
            else:
                self.engine.on_game_over(res[0], self.engine.color)
        QMessageBox.information(self, "Fine partita", "Esito: %s\nMotivo: %s" % (res[0], res[1]))
        self.update_panel()
        return True

    def undo_move(self):
        """Annulla la coppia di mosse (motore + umano), tornando al turno del giocatore."""
        if self._engine_thinking():
            return
        while len(self.history) > 1 and Board(self.history[-1]).turn != self.player_color:
            self.history.pop()
        if len(self.history) > 1:
            self.history.pop()
        self.board = Board(self.history[-1])
        self.selected = None
        self.legal_cache = []
        self._game_over = False
        self.redraw()
        self.update_panel()
        if self.board.turn != self.player_color and self.board.result() is None:
            self.status.setText("Turno del motore...")
            self._start_engine_move()
        else:
            self.status.setText("Mossa annullata. Tocca a te.")

    def _engine_thinking(self):
        return self.worker is not None and self.worker.isRunning()

    def _stop_worker(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(5000)
        self.worker = None

    # ---------- mapping display <-> scacchiera ----------

    def display_to_square(self, col, row):
        file = col if self.player_color == WHITE else 7 - col
        rank = 7 - row if self.player_color == WHITE else row
        if 0 <= file < 8 and 0 <= rank < 8:
            return rank * 8 + file
        return None

    def square_to_display(self, sq):
        col = FILES[sq] if self.player_color == WHITE else 7 - FILES[sq]
        row = 7 - RANKS[sq] if self.player_color == WHITE else RANKS[sq]
        return col, row

    # ---------- disegno ----------

    def redraw(self):
        scene = self.view.scene()
        scene.clear()
        scene.setSceneRect(0, 0, BOARD_PX + 2 * MARGIN, BOARD_PX + 2 * MARGIN)
        for sq in range(64):
            col, row = self.square_to_display(sq)
            x, y = MARGIN + col * SQ, MARGIN + row * SQ
            color = COLOR_LIGHT if (FILES[sq] + RANKS[sq]) % 2 == 0 else COLOR_DARK
            scene.addRect(QRectF(x, y, SQ, SQ), QPen(Qt.NoPen), QBrush(color))
        if self.last_move is not None:
            for s in self.last_move:
                col, row = self.square_to_display(s)
                scene.addRect(QRectF(MARGIN + col * SQ, MARGIN + row * SQ, SQ, SQ),
                              QPen(Qt.NoPen), QBrush(COLOR_LASTMOVE))
        if self.selected is not None:
            col, row = self.selected
            scene.addRect(QRectF(MARGIN + col * SQ, MARGIN + row * SQ, SQ, SQ),
                          QPen(Qt.NoPen), QBrush(COLOR_SELECTED))
        for m in self.legal_cache:
            col, row = self.square_to_display(m[1])
            cx = MARGIN + col * SQ + SQ / 2
            cy = MARGIN + row * SQ + SQ / 2
            r = SQ * 0.16 if self.board.board[m[1]] == 0 else SQ * 0.42
            scene.addItem(QGraphicsEllipseItem(cx - r, cy - r, 2 * r, 2 * r,
                                               QPen(Qt.NoPen), QBrush(COLOR_LEGAL)))
        for sq in range(64):
            p = self.board.board[sq]
            if p == 0:
                continue
            col, row = self.square_to_display(sq)
            t = p & TYPE_MASK
            glyph = GLYPH_WHITE[t] if (p & 8) == WHITE else GLYPH_BLACK[t]
            x = MARGIN + col * SQ
            y = MARGIN + row * SQ
            font = QFont("DejaVu Sans", int(SQ * 0.72))
            probe = QGraphicsTextItem(glyph)
            probe.setFont(font)
            w = probe.boundingRect().width()
            h = probe.boundingRect().height()
            px = x + (SQ - w) / 2
            py = y + (SQ - h) / 2
            if (p & 8) == WHITE:
                shadow = QGraphicsTextItem(glyph)
                shadow.setFont(font)
                shadow.setDefaultTextColor(QColor("black"))
                shadow.setPos(px + 1, py + 1)
                scene.addItem(shadow)
            item = QGraphicsTextItem(glyph)
            item.setFont(font)
            item.setDefaultTextColor(QColor("white") if (p & 8) == WHITE else QColor("black"))
            item.setPos(px, py)
            scene.addItem(item)
        # coordinate
        font = QFont("DejaVu Sans", 10)
        for c in range(8):
            file_label = chr(ord("a") + (c if self.player_color == WHITE else 7 - c))
            t1 = QGraphicsTextItem(file_label)
            t1.setFont(font)
            t1.setPos(MARGIN + c * SQ + SQ - 12, MARGIN + BOARD_PX + 6)
            scene.addItem(t1)
        for r in range(8):
            rank_label = str(8 - r) if self.player_color == WHITE else str(r + 1)
            t2 = QGraphicsTextItem(rank_label)
            t2.setFont(font)
            t2.setPos(MARGIN - 14, MARGIN + r * SQ + 4)
            scene.addItem(t2)

    # ---------- pannello informativo ----------

    def update_panel(self):
        if self.mode == "adaptive" and isinstance(self.engine, AdaptiveEngine):
            self.stack.setCurrentIndex(0)
            info = self.engine.calibration_info()
            self.lbl_rating.setText(
                "Rating stimato: %.0f (partite: %d)" % (info["rating"], info["games"])
            )
            self.lbl_calibrazione.setText(
                "Livello di calibrazione: profondità %d, prob. mossa sub-ottimale %.0f%%"
                % (info["profondita"], info["prob_subottimale"] * 100)
            )
        elif self.mode == "learner" and isinstance(self.engine, LearnerEngine):
            self.stack.setCurrentIndex(1)
            self.lbl_games.setText("Partite giocate: %d" % self.engine.games)
            for name, bar in self.weight_bars.items():
                w = self.engine.weights.get(name, 0.0)
                bar.setValue(max(-100, min(100, int(round(w * 100)))))
