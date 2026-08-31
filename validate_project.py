from pathlib import Path
import ast, json

root = Path(__file__).parent
for p in (root / "backend" / "app").glob("*.py"):
    ast.parse(p.read_text(encoding="utf-8"), filename=str(p))

json.loads((root / "mobile" / "app.json").read_text(encoding="utf-8"))
json.loads((root / "mobile" / "package.json").read_text(encoding="utf-8"))
json.loads((root / "mobile" / "eas.json").read_text(encoding="utf-8"))

print("Lio project static validation: OK")
