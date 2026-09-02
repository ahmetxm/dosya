const $ = (id) => document.getElementById(id);

function money(value) {
  const n = Number(value || 0);
  return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}¢`;
}

function cls(n) {
  return Number(n) >= 0 ? "up" : "down";
}

async function api(path, method = "GET") {
  const res = await fetch(path, { method });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

function renderStats(state) {
  const paper = state.paper || {};
  const spots = state.spots || {};
  const cards = [
    ["Durum", state.running ? "CANLI" : "DURDU"],
    ["BTC", spots.BTC ? money(spots.BTC) : "—"],
    ["ETH", spots.ETH ? money(spots.ETH) : "—"],
    ["Nakit", money(paper.cash)],
    ["Özkaynak", money(paper.equity)],
    ["P&L", money(paper.pnl)],
    ["Tur", state.cycle || 0],
    ["Taranan", state.scanned_markets || 0],
  ];
  $("stats").innerHTML = cards
    .map(([label, value], i) => {
      const extra = label === "P&L" ? cls(paper.pnl) : "";
      return `<div class="stat"><span>${label}</span><b class="${extra}">${value}</b></div>`;
    })
    .join("");
}

function renderOpps(rows) {
  if (!rows || !rows.length) {
    $("opps").textContent = "Bu turda eşik üstü fırsat yok.";
    return;
  }
  $("opps").innerHTML = `
    <table>
      <tr><th>Tür</th><th>Soru</th><th>Pay</th><th>Kenar</th><th>P&L</th></tr>
      ${rows
        .map(
          (row) => `<tr>
            <td><span class="tag ${row.locked ? "" : "warn"}">${row.kind}${row.locked ? "" : " · açık"}</span></td>
            <td>${row.question || row.event_title}</td>
            <td>${Number(row.shares).toFixed(1)}</td>
            <td class="up">${pct(row.edge_per_share)}</td>
            <td class="up">${money(row.expected_pnl)}</td>
          </tr>`
        )
        .join("")}
    </table>`;
}

function renderMisses(rows) {
  if (!rows || !rows.length) {
    $("misses").textContent = "Kitap verisi yok.";
    return;
  }
  $("misses").innerHTML = `
    <table>
      <tr><th>Soru</th><th>YES ask</th><th>NO ask</th><th>Toplam</th></tr>
      ${rows
        .map(
          (row) => `<tr>
            <td>${row.question}</td>
            <td>${row.yes_ask ?? "—"}</td>
            <td>${row.no_ask ?? "—"}</td>
            <td class="${row.pair_ask < 1 ? "up" : ""}">${row.pair_ask != null ? Number(row.pair_ask).toFixed(4) : "—"}</td>
          </tr>`
        )
        .join("")}
    </table>`;
}

function renderTrades(rows) {
  if (!rows || !rows.length) {
    $("trades").textContent = "Henüz kâğıt işlem yok.";
    return;
  }
  $("trades").innerHTML = `
    <table>
      <tr><th>Saat</th><th>Tür</th><th>P&L≈</th><th>Detay</th></tr>
      ${[...rows]
        .reverse()
        .map(
          (row) => `<tr>
            <td>${(row.ts || "").slice(11, 19)}</td>
            <td>${row.opportunity_kind}</td>
            <td class="${cls(row.expected_pnl)}">${money(row.expected_pnl)}</td>
            <td>${row.reason}</td>
          </tr>`
        )
        .join("")}
    </table>`;
}

function renderPositions(rows) {
  if (!rows || !rows.length) {
    $("positions").textContent = "Açık pozisyon yok.";
    return;
  }
  $("positions").innerHTML = `
    <table>
      <tr><th>Soru</th><th>Sonuç</th><th>Pay</th><th>Ort</th><th>Mid</th><th>UPL</th></tr>
      ${rows
        .map(
          (row) => `<tr>
            <td>${row.question}</td>
            <td>${row.outcome}</td>
            <td>${Number(row.shares).toFixed(2)}</td>
            <td>${Number(row.avg_price).toFixed(3)}</td>
            <td>${Number(row.mid).toFixed(3)}</td>
            <td class="${cls(row.unrealized)}">${money(row.unrealized)}</td>
          </tr>`
        )
        .join("")}
    </table>`;
}

function render(state) {
  renderStats(state);
  renderOpps(state.opportunities);
  renderMisses(state.near_misses);
  renderTrades((state.paper || {}).trades);
  renderPositions((state.paper || {}).positions);
  $("log").textContent = (state.log || []).join("\n") || (state.last_error || "hazır");
  if (state.last_error) $("log").textContent += `\nHATA: ${state.last_error}`;
}

async function refresh() {
  const state = await api("/api/state");
  render(state);
}

function bind(id, path) {
  $(id).addEventListener("click", async () => {
    $(id).disabled = true;
    try {
      const state = path === "/api/cycle" ? (await api(path, "POST")).state : await api(path, "POST");
      render(state);
    } catch (err) {
      $("log").textContent = String(err);
    } finally {
      $(id).disabled = false;
    }
  });
}

bind("startBtn", "/api/start");
bind("stopBtn", "/api/stop");
bind("resetBtn", "/api/reset");
bind("cycleBtn", "/api/cycle");
refresh();
setInterval(refresh, 2500);
