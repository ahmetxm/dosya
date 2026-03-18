# nekros_auto.py
import time
import json
import requests
from datetime import datetime
import threading
import websocket
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, ApiCreds
from py_clob_client.order_builder.constants import BUY, SELL
from dotenv import load_dotenv
import os

load_dotenv()

class NekrosAuto:
    def __init__(self, dry_run=True):
        self.host = "https://clob.polymarket.com"
        self.chain_id = 137  # Polygon Mainnet
        self.private_key = os.getenv("WALLET_PRIVATE_KEY")

        if not self.private_key:
            raise ValueError("Lütfen .env dosyasına WALLET_PRIVATE_KEY ekle! (Polygon cüzdan private key'i)")

        # Client'ı private key ile başlat
        self.client = ClobClient(
            host=self.host,
            key=self.private_key,
            chain_id=self.chain_id,
            signature_type=0,  # 0 = EOA (normal cüzdan), eğer Magic/email ise 1 yap
            # funder="<funder_address>" eğer proxy wallet kullanıyorsan ekle, yoksa kaldır
        )

        # API creds otomatik derive et ve set et (en kritik adım!)
        creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(creds)

        self.dry_run = dry_run
        self.market_slug_prefix = "bitcoin-up-down-5-minute"  # slug'ı güncel tut (siteye bak)
        self.up_token_id = None
        self.down_token_id = None
        self.current_up_price = 0.50
        self.balance = 1000.0  # başlangıç, sonra güncellenir
        self.ruh_sayisi = 0
        self.yara_sayisi = 0
        self.is_dead = False

        self.update_balance()
        self.find_active_btc_5min_market()

    def fisilt(self, msg):
        print(f"\033[90m{msg}\033[0m")  # gri/koyu

    def durum_raporu(self):
        can_yuzde = (self.balance / 1000.0) * 100 if self.balance > 0 else 0
        print("\n" + "═" * 60)
        print(f" N E K R O S   AUTO   {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Kalan kan:     {self.balance:.2f} USDC ({can_yuzde:.1f}%)")
        print(f"  Ruhlar:        {self.ruh_sayisi}")
        print(f"  Yaralar:       {self.yara_sayisi}")
        print(f"  Dry-run:       {'Açık (simülasyon)' if self.dry_run else 'Kapalı (GERÇEK TRADE)'}")
        print("═" * 60 + "\n")

    def update_balance(self):
        try:
            # Gerçek bakiye çek (USDC varsayıyoruz)
            bal_info = self.client.get_balance()
            self.balance = float(bal_info.get('usdc', {}).get('balance', 1000.0))
        except Exception as e:
            self.fisilt(f"Bakiye güncelleme hatası: {e} → Eski değer korunuyor.")

    def find_active_btc_5min_market(self):
        try:
            url = "https://gamma-api.polymarket.com/markets?limit=20&active=true&order_by=volume&ascending=false"
            resp = requests.get(url, timeout=10)
            markets = resp.json()
            for m in markets:
                slug = m.get('slug', '').lower()
                if "bitcoin" in slug and "up" in slug and "down" in slug and "5" in slug:
                    self.up_token_id = m['clobTokenIds'][0]   # Genelde Yes = Up
                    self.down_token_id = m['clobTokenIds'][1] # No = Down
                    self.fisilt(f"Aktif BTC 5min market bulundu: {m['question']}")
                    self.fisilt(f"Up Token ID: {self.up_token_id}")
                    return
            self.fisilt("BTC 5min market bulunamadı. Siteye girip token ID'yi manuel kodla.")
        except Exception as e:
            self.fisilt(f"Market arama hatası: {e}")

    def get_latest_up_price(self):
        try:
            ob = self.client.get_order_book(self.up_token_id)
            best_ask = float(ob.asks[0].price) if ob.asks else 0.50
            self.current_up_price = best_ask
            self.fisilt(f"UP güncel fiyat (best ask): {self.current_up_price:.4f}")
        except Exception as e:
            self.fisilt(f"Fiyat çekme hatası: {e} → Eski fiyat korunuyor.")

    def karar_ver(self):
        edge = self.current_up_price - 0.5
        if edge > 0.015:
            return "UP", self.up_token_id, BUY
        elif edge < -0.015:
            return "DOWN", self.down_token_id, SELL  # DOWN için No token alıyoruz (SELL mantığına dikkat)
        else:
            return None, None, None

    def pozisyon_ac(self, yon, token_id, side):
        if not token_id:
            self.fisilt("Token ID yok → trade atlanıyor.")
            return

        self.get_latest_up_price()

        miktar_str = input(f"\nNEKROS: {yon} yönünde karar verdi. Ne kadar USDC yatırmak istiyorsun? (max {self.balance:.2f}): ")
        try:
            miktar = float(miktar_str.strip())
            if miktar <= 0 or miktar > self.balance:
                self.fisilt("Geçersiz miktar → trade atlandı.")
                return
        except:
            self.fisilt("Geçersiz giriş → trade atlandı.")
            return

        price = self.current_up_price if yon == "UP" else (1 - self.current_up_price)
        share_size = miktar / price

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=share_size,
            side=side
        )

        self.fisilt(f"Pozisyon hazırlanıyor → {yon} | {miktar:.2f} USDC ≈ {share_size:.4f} share @ {price:.4f}")

        if self.dry_run:
            self.fisilt("DRY-RUN MODU: Gerçek order gönderilmedi (simülasyon).")
            kazandi = input("Simülasyon sonucu (kazandı mı? e / h): ").lower().strip() == 'e'
            self.trade_sonuc(kazandi, miktar)
        else:
            try:
                signed_order = self.client.create_order(order_args)
                resp = self.client.post_order(signed_order, order_type=OrderType.GTC)
                self.fisilt(f"Order başarıyla gönderildi: {resp}")
                # Gerçek sonuç için polling veya WS ekleyebilirsin
            except Exception as e:
                self.fisilt(f"Order gönderme hatası: {e}")

    def trade_sonuc(self, kazandi, miktar):
        if kazandi:
            kar = miktar * 0.98  # yaklaşık ücret sonrası
            self.balance += kar
            self.ruh_sayisi += 1
            self.fisilt(f"Zafer... bir ruh daha toplandı +{kar:.2f}")
        else:
            self.balance -= miktar
            self.yara_sayisi += 1
            self.fisilt(f"Yara açıldı... -{miktar:.2f} kan kaybı")

        if self.balance <= 5:
            self.is_dead = True
            self.fisilt("Kan tükendi... Nekros karanlığa gömüldü.")

    def run(self):
        print("\nNekros AUTO başladı. Karanlık devrede...")
        self.durum_raporu()

        while not self.is_dead:
            yon, token_id, side = self.karar_ver()
            if yon:
                self.pozisyon_ac(yon, token_id, side)
            else:
                self.fisilt("Edge yetersiz... izliyorum (bir sonraki döngüde tekrar bakılacak).")
            self.durum_raporu()
            time.sleep(60)  # her 60 sn kontrol et

if __name__ == "__main__":
    # İlk başta dry_run=True ile test et! False yaparsan GERÇEK PARA gider.
    nekros = NekrosAuto(dry_run=True)
    nekros.run()