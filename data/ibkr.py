"""
IBKR data connector — fetches spot VIX and VIX futures prices via ib_insync.
Requires IB Gateway running locally with API enabled.

Runs as a subprocess so ib_insync gets a clean main-thread event loop,
avoiding conflicts with Streamlit's async threading model.
"""

import json
import subprocess
import sys

_FETCH_SCRIPT = """
import asyncio
import json
import sys
from datetime import datetime

# Force a new event loop before ib_insync is imported (required on Python 3.10+)
asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Index, Future

host      = sys.argv[1]
port      = int(sys.argv[2])
client_id = int(sys.argv[3])

# In read-only mode reqExecutionsAsync never responds and causes connect() to
# time out. Patch it to return immediately — we don't need execution data.
async def _noop_executions(self, execFilter=None):
    return []
IB.reqExecutionsAsync = _noop_executions

result = {"spot": None, "futures": [], "timestamp": None, "error": None}
ib = IB()

try:
    ib.connect(host, port, clientId=client_id, timeout=15, readonly=True)

    import math

    def valid(v):
        return v is not None and not (isinstance(v, float) and math.isnan(v))

    def best_price(ticker):
        for v in [ticker.last, ticker.close, ticker.bid]:
            if valid(v):
                return float(v)
        return None

    # Spot VIX
    vix_contract = Index("VIX", "CBOE")
    ib.qualifyContracts(vix_contract)
    ticker = ib.reqMktData(vix_contract, "", False, False)
    ib.sleep(2)
    spot = best_price(ticker)

    # If market closed, fall back to last daily close from historical data
    if spot is None:
        bars = ib.reqHistoricalData(
            vix_contract, endDateTime="", durationStr="5 D",
            barSizeSetting="1 day", whatToShow="TRADES",
            useRTH=True, formatDate=1)
        if bars:
            spot = float(bars[-1].close)
            result["note"] = f"Market closed — using last close ({bars[-1].date})"

    result["spot"] = spot

    # VIX Futures
    fut_template = Future("VIX", exchange="CFE", currency="USD")
    details = ib.reqContractDetails(fut_template)
    contracts = [d.contract for d in details]

    if contracts:
        tickers = [ib.reqMktData(c, "", False, False) for c in contracts]
        ib.sleep(2)

        today = datetime.today()
        futures = []
        for contract, t in zip(contracts, tickers):
            expiry_raw = contract.lastTradeDateOrContractMonth
            try:
                expiry_dt = datetime.strptime(expiry_raw, "%Y%m%d")
            except ValueError:
                expiry_dt = datetime.strptime(expiry_raw, "%Y%m")

            tau   = (expiry_dt - today).days / 365.0
            price = best_price(t)

            # Fall back to historical close if no live price
            if price is None:
                bars = ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="2 D",
                    barSizeSetting="1 day", whatToShow="TRADES",
                    useRTH=True, formatDate=1)
                if bars:
                    price = float(bars[-1].close)

            if tau > 0 and price:
                futures.append({
                    "expiry": expiry_dt.strftime("%d %b %Y"),
                    "tau":    round(tau, 4),
                    "price":  round(price, 4),
                })

        result["futures"] = sorted(futures, key=lambda x: x["tau"])

    result["timestamp"] = datetime.now().strftime("%H:%M:%S")

except Exception as e:
    result["error"] = str(e)

finally:
    if ib.isConnected():
        ib.disconnect()

print(json.dumps(result))
"""


def fetch_vix_data(host: str = "127.0.0.1", port: int = 4001,
                   client_id: int = 10) -> dict:
    """
    Fetch spot VIX and VIX futures from IBKR by running a subprocess.

    Returns
    -------
    dict with keys: spot, futures, timestamp, error
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _FETCH_SCRIPT, host, str(port), str(client_id)],
            capture_output=True,
            text=True,
            timeout=40,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {
                "spot": None, "futures": [], "timestamp": None,
                "error": proc.stderr.strip() or "No output from subprocess.",
                "_debug": {"returncode": proc.returncode,
                           "stdout": proc.stdout, "stderr": proc.stderr,
                           "executable": sys.executable},
            }
        result = json.loads(proc.stdout.strip())
        result["_debug"] = {"returncode": proc.returncode, "stderr": proc.stderr,
                            "executable": sys.executable}
        return result

    except subprocess.TimeoutExpired:
        return {"spot": None, "futures": [], "timestamp": None,
                "error": "Connection timed out after 40 seconds."}
    except Exception as e:
        import traceback
        return {"spot": None, "futures": [], "timestamp": None,
                "error": str(e), "_debug": {"traceback": traceback.format_exc()}}
