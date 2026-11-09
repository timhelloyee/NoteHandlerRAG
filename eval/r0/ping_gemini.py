"""Is gemini-3.7-flash / 3.8-flash answering yet? Run by a UserPromptSubmit hook (asyncRewake).

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/ping_gemini.py

Pings each model not yet run on R0 (no results/model_g37flash or model_g38flash folder)
with a one-word prompt, both keys, 20 s each. Exit 2 with the run command on stderr when one
answers, so the hook wakes Claude to start it; exit 0 quietly otherwise. Once both folders
exist the script does nothing.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MODELS = {"gemini-3.7-flash": "model_g37flash", "gemini-3.8-flash": "model_g38flash"}


def alive(model: str) -> bool:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    load_dotenv(ROOT / ".env")
    for key in ("GEMINI_API_KEY_SECONDARY", "GEMINI_API_KEY_PRIMARY"):
        try:
            c = genai.Client(api_key=os.environ[key], http_options=types.HttpOptions(timeout=20_000))
            if c.models.generate_content(model=model, contents="回一個字：好").text:
                return True
        except Exception:                            # noqa: BLE001 - any failure = not available
            pass
    return False


def main():
    todo = [m for m, d in MODELS.items() if not (HERE / "results" / d).exists()]
    if not todo:
        return
    with ThreadPoolExecutor(len(todo)) as pool:
        up = [m for m, ok in zip(todo, pool.map(alive, todo)) if ok]
    if up:
        cmds = "\n".join(f"  PYTHONIOENCODING=utf-8 C:/Users/user/anaconda3/envs/ragtraining/python.exe "
                         f"eval/r0/run_baseline.py --model {m} --out eval/r0/results/{MODELS[m]}" for m in up)
        print(f"[Gemini ping] 可用且尚未跑過 R0：{'、'.join(up)}。使用者已授權直接在背景執行"
              f"（在 C:/Users/user/Desktop/ClaudeProjects/NoteHandler）：\n{cmds}\n"
              "跑完用 compare_runs.py baseline <資料夾> 評分並回報。", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
