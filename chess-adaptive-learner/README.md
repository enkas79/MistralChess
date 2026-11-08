# Scacchi Adattivo / Imparante

App desktop di scacchi contro il computer con **due motori selezionabili**, entrambi
autoimplementati in Python puro: nessuna libreria scacchistica (niente `python-chess`),
nessun engine esterno (niente Stockfish), nessun dato scaricato. L'unica dipendenza
esterna è **PySide6** per la GUI.

## Installazione e avvio

```bash
pip install -r requirements.txt
python main.py
```

Richiede Python 3.10+ e una sessione grafica (oppure `QT_QPA_PLATFORM=offscreen` per
test senza display).

## Struttura

```
chess-adaptive-learner/
├── README.md
├── .gitignore                # __pycache__, *.json di stato
├── requirements.txt          # solo: PySide6
├── main.py                   # avvio GUI
├── src/
│   ├── engine.py             # regole scacchi complete (mailbox 64, make/unmake)
│   ├── learner.py            # motore che impara (16 feature + regressione logistica)
│   ├── adaptive.py           # motore che si calibra sul livello dell'umano
│   └── gui.py                # PySide6 (Qt): QGraphicsView + QThread per il motore
└── tests/
    ├── test_engine.py        # perft + casi speciali
    └── test_learning.py      # convergenza pesi + partite simulate
```

## Le due modalità

### Motore Adattivo (`src/adaptive.py`)
- Valutazione fissa solida: materiale + tavole posizionali (PST classiche) + mobilità.
- Ricerca negamax con alpha-beta, profondità regolabile 1–3.
- Dopo ogni mossa dell'umano calcola la **perdita in centipedi** rispetto alla
  migliore mossa (ricerca di riferimento a profondità 3) e la accumula.
- A fine partita: `rating = media mobile di (2000 − 45 × perdita_media)`,
  corretto di ±50 in base al risultato, con clamp 0–2000. Stato salvato in
  `adaptive_state.json` (gitignorato).
- Gioca con profondità in funzione del rating stimato; con probabilità decrescente
  col rating sceglie la 2ª o 3ª mossa migliore invece della migliore.
- Il pannello mostra **rating stimato** e **livello di calibrazione**.

### Motore Imparante (`src/learner.py`)
- Sa solo le regole all'inizio: la valutazione è una combinazione lineare di
  **16 feature** (materiale ×5, mobilità, coppia degli alfieri, pedoni doppiati,
  pedoni isolati, pedoni passati, scudo di pedoni del re, controllo del centro,
  torri su colonna aperta, pressione sul re nemico, sviluppo, avanzamento pedoni)
  con pesi iniziali casuali in [-0.5, 0.5].
- Ricerca negamax con alpha-beta, profondità 2–3, ordinamento mosse leggermente casuale.
- A fine partita aggiorna i pesi con **regressione logistica** verso l'esito
  (1 vittoria / 0.5 patta / 0 sconfitta): `w += lr * (esito - sigmoid(w·f)) * f`
  con `lr = 0.02`. Pesi e contatore partite salvati in `learner_weights.json` (gitignorato).
- Il pannello mostra i **pesi appresi con barre +/-**.

## GUI (`src/gui.py`)

- Finestra principale con scacchiera in `QGraphicsView`/`QGraphicsScene`: caselle
  cliccabili, evidenziazione della selezione e delle mosse legali, coordinate,
  pezzi Unicode.
- Pannello laterale: selezione della modalità (radio), colore del giocatore,
  pulsanti **Nuova partita** e **Annulla mossa** (annulla la coppia di mosse),
  area informativa con rating/pesi appresi.
- Il calcolo della mossa del motore avviene in un **QThread dedicato**: quando pronto
  emette un `Signal` e la GUI aggiorna la scena; nessuna chiamata diretta ai widget
  dal thread di lavoro.
- Promozione sempre a dama (prototipo).
- A fine partita: `QMessageBox` con l'esito, poi aggiornamento di apprendimento/rating.

## Test

```bash
python -m pytest tests/ -q          # perft fino a d4 + casi speciali + convergenza pesi
PERFT_D5=1 python -m pytest tests/test_engine.py -q   # include anche perft d5 (4.865.609)
```

Validazione perft dalla posizione iniziale: 20 (d1), 400 (d2), 8.902 (d3),
197.281 (d4), 4.865.609 (d5, opzionale). Test extra su Kiwipete e posizioni
con en passant/promozioni/arrocco.

## Limiti dichiarati

- Il **motore imparante** piattizza intorno a un livello equivalente di circa
  1000–1200 dopo le prime decine di partite e **overfitta lo stile dell'unico
  avversario umano** con cui si allena.
- La **stima di forza del motore adattivo** è indicativa e non è basata su Stockfish
  né su altri engine di riferimento.
- La ricerca è negamax alpha-beta a profondità 2–3 senza quiescence: il motore può
  commettere errori tattici (orizzonte'orizzonte).
- Non è implementata la regola della triplice ripetizione (sono coperti scacco matto,
  stallo, regola delle 50 mosse e materiale insufficiente).
