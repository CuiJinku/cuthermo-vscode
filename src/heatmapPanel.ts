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
        // allow comma or whitespace separated
        const toks = line.split(/(?:,|\s)+/).map(s => s.trim()).filter(Boolean);
        if (toks.length < 10) continue;
        const sectorTag = toks[0];
        const freqs = toks.slice(1, 10).map(x =>
            (x === '^' || x === 'v' || x === '.' || x === '|') ? x : Number(x)
        );
        if (sectorTag !== '0x...........' &&
            freqs.some(v => typeof v === 'number' && Number.isNaN(v))) continue;
        rows.push({ sectorTag, frequencies: freqs });
    }
    return rows;
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
        this.allRows = parseTxtHeatmap(txt);

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
  <meta charset="UTF-8"/>
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>cuThermo Heatmap</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 10px; }
    #canvas { border: 1px solid #ddd; display: block; }
    .legend { margin-top: 6px; font-size: 12px; opacity: 0.8; }
    /* Make page scroll naturally if the canvas is tall */
    html, body { height: 100%; }
  </style>
</head>
<body>
  <canvas id="canvas" width="900" height="300"></canvas>
  <div class="legend">Numbers colored; symbols (|, ^, v, .) shown in light gray. Scroll to view more rows.</div>
  <script>
    const COLORS = ["#3a4cc0","#6282ea","#f4997a","#6282ea","#f4997a","#f5c4ac","#f4997a","#dd5f4b","#b40326"];
    const CELL = 24;       // cell width
    const ROW_H = 24;      // row height
    const LEFT_W = 160;    // sector tag column width
    const COLS = 9;        // 8 W columns + 1 Total

    let rows = [];

    function drawAll() {
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');

      const headerH = 24;
      const totalH = headerH + ROW_H * rows.length + 12;
      const totalW = Math.max(LEFT_W + COLS * CELL + 20, 900);

      // Resize canvas to content height so the page scrolls naturally
      canvas.width = totalW;
      canvas.height = totalH;

      // header
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.fillStyle = '#222';
      ctx.font = 'bold 13px system-ui, sans-serif';
      ctx.fillText('Sector Tag', 8, 16);
      const headers = ['W0','W1','W2','W3','W4','W5','W6','W7','Total'];
      headers.forEach((h,i) => ctx.fillText(h, LEFT_W + i*CELL + 6, 16));

      // rows
      ctx.font = '12px system-ui, sans-serif';
      for (let ri=0; ri<rows.length; ri++) {
        const r = rows[ri];
        const y = headerH + (ri+1) * ROW_H - 4;

        // sector tag
        ctx.fillStyle = '#333';
        ctx.fillText(r.sectorTag, 8, y);

        // 9 cells
        for (let ci=0; ci<r.frequencies.length; ci++) {
          const v = r.frequencies[ci];
          const x = LEFT_W + ci * CELL;
          const isSym = (v === '|' || v === '^' || v === 'v' || v === '.' || v === '🡅' || v === '🡇');

          // background
          if (isSym) {
            ctx.fillStyle = '#dcdcdc';
          } else {
            const idx = Math.max(0, Math.min(8, Number(v)));
            ctx.fillStyle = COLORS[idx] || '#eee';
          }
          ctx.fillRect(x, y-ROW_H+6, CELL-2, ROW_H-8);

          // text
          ctx.fillStyle = '#111';
          let t = (v === '^') ? '🡅' : (v === 'v') ? '🡇' : String(v);
          ctx.fillText(t, x + 6, y - 6);
        }
      }
    }

    window.addEventListener('message', (e) => {
      const { type, payload } = e.data || {};
      if (type === 'data') {
        rows = payload.rows || [];
        drawAll();
      }
    });

    // Repaint on resize (e.g., zoom) so text remains crisp
    window.addEventListener('resize', () => drawAll());
  </script>
</body>
</html>`;
    }
}
