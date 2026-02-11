import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import time

start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Kullanılan cihaz: {device}')

# Veri hazırlama (öncekiyle aynı)
zigzag_df = pd.read_csv('ETHUSD_ZigZag_Result.csv')
price_df = pd.read_csv('ETHUSD.csv')

zigzag_df['Date'] = pd.to_datetime(zigzag_df['Date'])
price_df['Open time'] = pd.to_datetime(price_df['Open time'])

zigzag_df = zigzag_df.set_index('Date')
price_df = price_df.set_index('Open time')
merged_df = price_df.join(zigzag_df, how='left')

merged_df['Type'] = merged_df['Type'].fillna('None')
merged_df['Signal'] = 0
merged_df.loc[merged_df['Type'] == 'Low', 'Signal'] = 1
merged_df.loc[merged_df['Type'] == 'High', 'Signal'] = -1

data = merged_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
labels = merged_df['Signal'].values + 1

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

seq_length = 120
X, y = [], []
for i in range(len(data_scaled) - seq_length):
    X.append(data_scaled[i:i+seq_length])
    y.append(labels[i+seq_length])

X = np.array(X)
y = np.array(y, dtype=int)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).long().to(device)
X_test = torch.from_numpy(X_test).float().to(device)

class SignalDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(SignalDataset(X_train, y_train), batch_size=128, shuffle=True)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_size=100, num_layers=3, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(3, x.size(0), 100).to(device)
        c0 = torch.zeros(3, x.size(0), 100).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

model = LSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(30):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Batch prediction
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    signals = predicted.cpu().numpy() - 1

# ────────────────────────────────────────────────
#          BACKTEST - TRAILING STOP EKLENDİ
# ────────────────────────────────────────────────

initial_capital = 1000.0
leverage = 100
capital = initial_capital
positions = []
fees = 0.001

sl_percent = 0.02          # başlangıç stop-loss
tp_percent = 0.06          # take-profit
trailing_activation = 0.03 # %3 kar sonrası trailing başlar
trailing_distance = 0.025  # trailing mesafesi %2.5
max_drawdown = 0.50
liq_margin = 1.0 / leverage

test_start = split + seq_length
signal_idx = 0

for i in range(test_start, len(merged_df)):
    if capital < initial_capital * (1 - max_drawdown):
        print("Max drawdown aşıldı → trading durduruldu")
        break

    current_price = merged_df.iloc[i]['Close']
    current_high = merged_df.iloc[i]['High']   # trailing için high/low takip
    current_low  = merged_df.iloc[i]['Low']
    signal = signals[signal_idx] if signal_idx < len(signals) else 0
    signal_idx += 1

    position_size = min(100.0, capital * 0.1)

    if positions:
        pos = positions[0]
        close_pos = False
        pnl_ratio = 0.0

        if pos['type'] == 'long':
            pnl_ratio = (current_price - pos['entry']) / pos['entry']

            # Likidasyon kontrolü
            if pnl_ratio <= -liq_margin:
                capital -= pos['margin']
                close_pos = True
                print(f"Liquidation LONG @ {current_price:.2f}")

            # Normal SL
            elif pnl_ratio <= -sl_percent:
                close_pos = True

            # Take-profit
            elif pnl_ratio >= tp_percent:
                close_pos = True

            # Trailing stop
            else:
                # En yüksek seviyeyi güncelle
                if current_high > pos['max_price']:
                    pos['max_price'] = current_high

                # Trailing aktif mi?
                if pnl_ratio >= trailing_activation:
                    trailing_stop = pos['max_price'] * (1 - trailing_distance)
                    if current_price <= trailing_stop:
                        close_pos = True
                        print(f"Trailing stop tetiklendi LONG @ {current_price:.2f} (max: {pos['max_price']:.2f})")

            # Model sinyali ters ise kapat
            if signal == -1:
                close_pos = True

        else:  # SHORT
            pnl_ratio = (pos['entry'] - current_price) / pos['entry']

            if pnl_ratio <= -liq_margin:
                capital -= pos['margin']
                close_pos = True
                print(f"Liquidation SHORT @ {current_price:.2f}")

            elif pnl_ratio <= -sl_percent:
                close_pos = True

            elif pnl_ratio >= tp_percent:
                close_pos = True

            else:
                if current_low < pos['min_price']:
                    pos['min_price'] = current_low

                if pnl_ratio >= trailing_activation:
                    trailing_stop = pos['min_price'] * (1 + trailing_distance)
                    if current_price >= trailing_stop:
                        close_pos = True
                        print(f"Trailing stop tetiklendi SHORT @ {current_price:.2f} (min: {pos['min_price']:.2f})")

            if signal == 1:
                close_pos = True

        if close_pos:
            pnl = pnl_ratio * pos['size']
            pnl -= pos['margin'] * fees
            capital += pos['margin'] + pnl
            positions = []

    # Yeni pozisyon açma
    if not positions and capital >= position_size:
        if signal == 1:  # LONG
            effective_size = position_size * leverage
            positions.append({
                'type': 'long',
                'entry': current_price,
                'size': effective_size,
                'margin': position_size,
                'max_price': current_price   # trailing için başlangıç
            })
            capital -= position_size + (position_size * fees)
        elif signal == -1:  # SHORT
            effective_size = position_size * leverage
            positions.append({
                'type': 'short',
                'entry': current_price,
                'size': effective_size,
                'margin': position_size,
                'min_price': current_price   # trailing için başlangıç
            })
            capital -= position_size + (position_size * fees)

# Son pozisyonu kapat
if positions:
    pos = positions[0]
    exit_price = merged_df.iloc[-1]['Close']
    if pos['type'] == 'long':
        pnl_ratio = (exit_price - pos['entry']) / pos['entry']
    else:
        pnl_ratio = (pos['entry'] - exit_price) / pos['entry']
    pnl = pnl_ratio * pos['size']
    pnl -= pos['margin'] * fees
    capital += pos['margin'] + pnl

print(f"\nSon Sermaye   : ${capital:,.2f}")
print(f"Getiri        : {capital / initial_capital:.1f}x")

print(f"Çalışma süresi: {time.time() - start_time:.1f} saniye")
