"""Word frequency in the Academia Sinica Balanced Corpus 4.0 (中研院平衡語料庫), through its free
online search (asbc.iis.sinica.edu.tw). The count is the number of hits the search reports for
the keyword. Answers are cached in results/sinica_freq.json and requests are spaced one second
apart, so the public server sees each string once. Cite CKIP, Academia Sinica, when used.
"""
import json
import re
import time
from pathlib import Path

import requests

CACHE = Path(__file__).resolve().parent / "results" / "sinica_freq.json"
BASE = "https://asbc.iis.sinica.edu.tw/"
_cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
_session = None


def freq(word: str) -> int:
    global _session
    if word not in _cache:
        if _session is None:
            _session = requests.Session()
            _session.post(BASE + "index2.asp", data=b"", timeout=30)
        time.sleep(1)
        r = _session.post(BASE + "process.asp", timeout=60, data={
            "inputword": word.encode("cp950"), "selectAB": "", "inputpos": "", "selectFeature": ""})
        text = re.sub(r"<[^>]+>", " ", r.content.decode("cp950", "replace"))
        m = re.search(r"共有\s*(\d+)\s*筆", text)
        if not m and "查詢結果" not in text:            # not a result page: don't cache a fake 0
            raise RuntimeError(f"{word}: unexpected page {re.sub(chr(92) + 's+', ' ', text)[:200]}")
        _cache[word] = int(m.group(1)) if m else 0
        CACHE.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return _cache[word]


if __name__ == "__main__":
    import sys
    for w in sys.argv[1:]:
        print(w, freq(w))
