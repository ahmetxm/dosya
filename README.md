# Polymarket crypto paper arb

Canlı Polymarket kripto piyasalarını tarayan **kâğıt ticaret** simulasyonu. Gerçek cüzdan, API anahtarı veya CLOB emri yok.

Kaynaklar (hepsi public, auth yok):

- Gamma ` /events?tag_id=21 ` — açık kripto eventleri
- CLOB ` /book ` — YES/NO derinlik
- Coinbase ` /products/BTC-USD|ETH-USD/ticker ` — spot

## Ne yakalar

1. **binary_buy_pair** — YES ask + NO ask + taker fee < $1. Çözümde çift $1 öder.
2. **binary_sell_pair** — YES bid + NO bid − fee > $1. İkisini de short etmek overround kilitler.
3. **strike_monotonicity** — Aynı merdivende zor YES, kolay YES’ten pahalı bid ediliyorsa buy-easy / sell-hard.
4. **spot_yes/no_certainty** — Vade yakınken spot strike’ı net geçmişse (kilitli değil, yüksek güven).

Taker fee resmi formül: `fee = C × feeRate × p × (1 − p)` (crypto `feeRate = 0.07`).  
[Polymarket fees](https://docs.polymarket.com/trading/fees)

## Çalıştır

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 -m sim --once
python3 -m sim --serve --port 8765
```

Dashboard: `http://127.0.0.1:8765`

`--locked-only` spot-certainty işlemlerini kapatır. Başlangıç nakit varsayılan $10,000 kâğıt USDC.

## Not

Bu bir simulasyon. Spread, kayma, fee ve çözülme riski canlı kitapta değişir; kâğıt P&L gerçek para değildir.
