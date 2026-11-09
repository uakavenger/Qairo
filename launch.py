"""Start QAIRO locally and open it in the default browser (standard library only)."""
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:5000"


def main():
    server = subprocess.Popen([sys.executable, str(ROOT / "local_server.py")], cwd=ROOT)
    try:
        for _ in range(60):
            if server.poll() is not None:
                raise SystemExit("Сервер іске қосылмады. Терминалдағы қатені қараңыз.")
            try:
                with urllib.request.urlopen(URL + "/api/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(.25)
        else:
            raise SystemExit("Сервер 15 секунд ішінде жауап бермеді.")
        print(f"QAIRO ашылды: {URL}\nТоқтату үшін Ctrl+C басыңыз.", flush=True)
        webbrowser.open(URL)
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
