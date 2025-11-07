// src/heatmapPanel.ts
import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import * as path from 'path';

type Row = { sectorTag: string; frequencies: (number | string)[] }; // 9 cells

function parseTxtHeatmap(txt: string): Row[] {
    const rows: Row[] = [];
    const lines = txt.split(/\r?\n/);
    for (const raw of lines) {
        const line = raw.trim();
        if (!line) continue;

        // Split by comma or whitespace
        const toks = line.split(/(?:,|\s)+/).map(s => s.trim()).filter(Boolean);
        if (toks.length < 10) continue;

        // Handle "Sector:" prefix + number
        let sectorTag = toks[0];
        if (sectorTag.toLowerCase() === "sector:" && toks[1]) {
            const maybeHex = toks[1].startsWith("0x") ? toks[1] : "0x" + toks[1];
            sectorTag = `Sector: ${maybeHex.toLowerCase()}`;
            toks.splice(0, 2); // remove first two tokens from frequency array
        }

        const freqs = toks.slice(0, 9).map(x =>
            (x === '^' || x === 'v' || x === '.' || x === '|') ? x : Number(x)
        );

        if (sectorTag !== '0x...........' &&
            freqs.some(v => typeof v === 'number' && Number.isNaN(v))) continue;

        rows.push({ sectorTag, frequencies: freqs });
    }
    return rows;
}

function collapseDuplicateRuns(input: Row[]): Row[] {
    const out: Row[] = [];
    let duplicateBuffer: Row[] = [];        // up to 10 rows
    let duplicateCount = 0;                 // # beyond the buffer
    let lastFreqKey: string | undefined;    // string key of the 9-tuple

    const keyOf = (freqs: (number | string)[]) => freqs.map(String).join('\u0001');

    const mockRow = (value: string): Row => ({
        sectorTag: '0x...........',
        frequencies: Array.from({ length: 9 }, (_, i) => (i === 5 ? value : '.'))
    });

    const flush = () => {
        if (!duplicateBuffer.length) return;
        // We were tracking a run of identical frequency rows (key = lastFreqKey).
        if (duplicateCount > 0) {
            // Collapsed form: first occurrence, ^, count, v, last occurrence
            const first = duplicateBuffer[0];
            const last = duplicateBuffer[duplicateBuffer.length - 1];
            // Ensure we output first and last with the repeated frequencies
            const freqs = first.frequencies;
            out.push({ sectorTag: first.sectorTag, frequencies: freqs });
            out.push(mockRow('^'));
            out.push(mockRow(String(duplicateCount)));
            out.push(mockRow('v'));
            out.push({ sectorTag: last.sectorTag, frequencies: freqs });
        } else {
            // < 11 total duplicates → output buffer verbatim
            out.push(...duplicateBuffer);
        }
        // reset
        duplicateBuffer = [];
        duplicateCount = 0;
    };

    for (const row of input) {
        const k = keyOf(row.frequencies);
        if (lastFreqKey === undefined || k === lastFreqKey) {
            // same as previous run (or first row)
            if (duplicateBuffer.length < 10) {
                duplicateBuffer.push(row);
            } else {
                // beyond the 10-row buffer: only update last sectorTag and count
                duplicateBuffer[duplicateBuffer.length - 1] = {
                    sectorTag: row.sectorTag,
                    frequencies: row.frequencies
                };
                duplicateCount += 1;
            }
        } else {
            // run breaks → flush previous run and start new
            flush();
            duplicateBuffer = [row];
        }
        lastFreqKey = k;
    }

    // final flush
    flush();

    return out;
}


export class HeatmapPanel {
    private panel: vscode.WebviewPanel | undefined;
    private allRows: Row[] = [];

    constructor(
        private readonly context: vscode.ExtensionContext,
        private readonly filePath: string
    ) { }

    async show() {
        const txt = await fs.readFile(this.filePath, 'utf8');
        const parsed = parseTxtHeatmap(txt);          // raw rows
        this.allRows = collapseDuplicateRuns(parsed); // 🔥 apply your Python logic


        const webview = this.createPanel().webview;
        webview.html = this.getHtml(webview);

        // 👉 Send ALL rows at once
        webview.postMessage({ type: 'data', payload: { rows: this.allRows } });

        // (Optional) still support manual reload trigger if you add such a button later
        webview.onDidReceiveMessage((msg) => {
            if (msg?.type === 'reload') this.reload();
        });
    }

    private async reload() {
        const txt = await fs.readFile(this.filePath, 'utf8');
        this.allRows = parseTxtHeatmap(txt);
        this.panel?.webview.postMessage({ type: 'data', payload: { rows: this.allRows } });
    }
    private slice(start: number, end: number) {
        return {
            start, end, total: this.allRows.length,
            rows: this.allRows.slice(start, end)
        };
    }


    private createPanel() {
        if (this.panel) {
            this.panel.reveal(vscode.ViewColumn.Active);
            return this.panel;
        }
        this.panel = vscode.window.createWebviewPanel(
            'cuthermoHeatmap',
            `Heatmap: ${path.basename(this.filePath)}`,
            vscode.ViewColumn.Active,
            { enableScripts: true, retainContextWhenHidden: true }
        );
        this.panel.onDidDispose(() => { this.panel = undefined; });
        return this.panel;
    }

    private getHtml(webview: vscode.Webview) {
        const csp = `default-src 'none'; style-src 'unsafe-inline' ${webview.cspSource}; script-src 'unsafe-inline' ${webview.cspSource}; img-src ${webview.cspSource} data:;`;
        return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>cuThermo Heatmap</title>
  <style>
/* ==== Your CSS (slightly tweaked: allow page to scroll) ==== */

/* Ensure full height usage */
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  overflow: hidden;
  font-family: "Courier New", Courier, monospace;
}

.container {
  width: 100%;
  height: 100%;
  overflow-y: hidden;
  display: flex;
  flex-direction: column;
}


/* Header Row */
.headerRow {
  display: flex;
  position: sticky;
  top: 0;
  color: black;
  font-weight: bold;
  text-align: center;
  border-bottom: 2px solid #000;
  z-index: 5;
}

/* Keep header cells aligned properly */
.headerCell {
  width: 40px;  /* 🔥 1cm width */
  height: 40px; /* 🔥 1cm height */
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  border: 1px solid #ddd;
  color: white;
  background: #386a1f;
}

/* Increase width of the first column ("Sector Tag") */
.sectorTagHeader {
  width: 220px !important;
  display: flex;
  align-items: center;
  justify-content: flex-start;  /* left-align */
  padding-left: 8px;            /* small left padding */
}

/* Center the whole heatmap horizontally but keep it at the top */
.centerContainer {
  display: flex;
  justify-content: center;
  align-items: flex-start;  /* keep top-aligned */
  flex: 1;                  /* allow it to expand */
  width: 100%;
  height: 100%;
}

/* Ensure the table stays at the top */
.tableWrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;                  /* Let it grow to fill container */
  height: 100%;             /* Use full height */
  max-height: 100vh;        /* Cap at viewport height just in case */
  overflow-y: auto;
  border: 1px solid #ccc;
  background: #ffffff;
}

/* Loading indicator */
.loadingText {
  text-align: center;
  padding: 10px;
  font-size: 14px;
  font-weight: bold;
}

/* Ensure the heatmap remains inside the centered layout */
.heatmap {
  position: relative;
  width: auto;
  flex-grow: 1;
}

/* Ensure each row stays in a single line */
.heatmapRow {
  display: flex;
  width: 100%;
  align-items: center;
}

/* Fix sector tag alignment (first column in data) */
.sectorTag {
  width: 220px !important;  /* match header */
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: flex-start;  /* left-align to remove big gap */
  padding-left: 8px;
  font-size: 14px;
  font-weight: bold;
  border: 1px solid #ddd;
  background-color: #b8f397;
  color: black;
  text-align: left;
  font-family: "Courier New", Courier, monospace;
}

/* Fix heatmap cell layout */
.heatmapCell {
  width: 40px;   /* 🔥 Keep cells 1cm wide */
  height: 40px;  /* 🔥 Keep cells 1cm high */
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: bold;
  color: black;
  border: 1px solid #ddd;
  transition: background-color 0.2s ease-in-out;
}

/* 🔥 Remove Grey Background for Mock Rows (container row can be styled) */
.mockRow {
  background-color: transparent !important;
}

/* 🔥 Ensure the mock symbols still have proper spacing */
.mockCell {
  font-style: italic;
  color: gray;
}

/* 🔥 Add space before the last column in the header */
.lastSecondHeaderCell {
  margin-right: 10px;  /* 🔥 Matches the spacing in .lastSecondCell */
}

/* 🔥 Add space before the last column */
.lastSecondCell {
  margin-right: 10px;  /* 🔥 Adjust this value to control spacing */
}
  </style>
</head>
<body>
<div class="container">
  <div class="centerContainer">
    <div class="tableWrapper">
      <!-- Header -->
      <div class="headerRow" id="headerRow">
        <div class="headerCell sectorTagHeader">Sector Tag</div>
        <div class="headerCell">W0</div>
        <div class="headerCell">W1</div>
        <div class="headerCell">W2</div>
        <div class="headerCell">W3</div>
        <div class="headerCell">W4</div>
        <div class="headerCell">W5</div>
        <div class="headerCell">W6</div>
        <div class="headerCell lastSecondHeaderCell">W7</div>
        <div class="headerCell">Total</div>
      </div>

      <!-- Heatmap body -->
      <div class="heatmap" id="heatmap"></div>

      <div class="loadingText" id="footerMsg" style="display:none;">No more data</div>
    </div>
  </div>
</div>

  <script>
    // VS Code messaging

    const vscode = acquireVsCodeApi();

    // Lighter palette (you can tweak to taste)
    // If your numeric values are 0..8, this maps directly
    const COLORS = [
      "#2c7bb6",
      "#00a6ca",
      "#00ccbc",
      "#90eb9d",
      "#ffff8c",
      "#f9d057",
      "#f29e2e",
      "#e76818",
      "#d7191c"
    ];
    const NEUTRAL = "#e6e6e6";
    
    function colorFor(value, colIndex, isTotalCol = false) {
      if (value === "|" || value === "🡅" || value === "🡇" || value === ".") return NEUTRAL;
      if (isTotalCol) return NEUTRAL;
    
      const idx = Number(value);
      if (!Number.isFinite(idx) || idx < 0 || idx >= COLORS.length) return NEUTRAL;
      return COLORS[idx];
    }

    // Optional: brighten/dim any hex color by 'factor' (0..1 mix with white)
    function lighten(hex, factor = 0.2) {
      const v = hex.replace('#','');
      const r = parseInt(v.slice(0,2),16), g = parseInt(v.slice(2,4),16), b = parseInt(v.slice(4,6),16);
      const mix = (c) => Math.round(c + (255 - c) * factor);
      return '#' + [mix(r), mix(g), mix(b)].map(x => x.toString(16).padStart(2,'0')).join('');
    }

    function renderAll(rows) {
      const root = document.getElementById('heatmap');
      root.innerHTML = '';

      const frag = document.createDocumentFragment();

      for (const row of rows) {
        const isMock = (row.sectorTag === '0x...........');
        const r = document.createElement('div');
        r.className = 'heatmapRow' + (isMock ? ' mockRow' : '');

        const tag = document.createElement('div');
        tag.className = 'sectorTag';
        tag.textContent = row.sectorTag.trim();
        r.appendChild(tag);

        row.frequencies.forEach((v, idx) => {
          const cell = document.createElement('div');
          cell.className = 'heatmapCell' + (idx === 7 ? ' lastSecondCell' : '');

          // symbols mapping (for display only)
          let display = v;
          if (v === '^') display = '🡅';
          if (v === 'v') display = '🡇';
          
          const isTotalCol = (idx === row.frequencies.length - 1);
          cell.style.backgroundColor = colorFor(display, idx, isTotalCol);
          cell.textContent = String(display);
          r.appendChild(cell);
        });

        frag.appendChild(r);
      }

      root.appendChild(frag);
    }

    // Handle data from extension
    window.addEventListener('message', (e) => {
      const { type, payload } = e.data || {};
      if (type === 'data') {
        const rows = payload?.rows || [];
        renderAll(rows);
        document.getElementById('footerMsg').style.display = 'none';
      }
    });
  </script>
</body>
</html>`;
    }

}
