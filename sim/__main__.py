"""CLI: python -m sim --serve   or   python -m sim --live-seconds 45"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .engine import EngineConfig, LiveEngine
from .server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Polymarket crypto paper-trading arb simulator")
    parser.add_argument("--serve", action="store_true", help="open the live dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--live-seconds", type=float, default=0, help="run the live loop then exit")
    parser.add_argument("--once", action="store_true", help="one live scan cycle then print JSON")
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--events", type=int, default=10)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--locked-only", action="store_true", help="skip unlocked spot-certainty trades")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = LiveEngine(
        config=EngineConfig(
            starting_cash=args.cash,
            event_limit=args.events,
            interval_sec=args.interval,
            allow_unlocked=not args.locked_only,
        )
    )

    if args.once:
        result = engine.run_cycle()
        json.dump(engine.snapshot() | {"cycle_result": result}, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    if args.live_seconds > 0:
        engine.start()
        engine.run_for(args.live_seconds)
        engine.stop()
        json.dump(engine.snapshot(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    if args.serve:
        engine.start()
        server = serve(engine, host=args.host, port=args.port)
        print(f"paper dashboard http://127.0.0.1:{args.port}  (simulation only, no live orders)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            engine.stop()
            server.shutdown()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
