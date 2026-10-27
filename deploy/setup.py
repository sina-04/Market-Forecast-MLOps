"""Generate local secrets without printing them. Uses only the Python standard library."""
import os
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[1]
path = root / ".env"
if path.exists():
    print("Existing .env preserved.")
else:
    path.write_text("MARKETFORECAST_API_KEY=" + secrets.token_urlsafe(32) + "\nGRAFANA_ADMIN_PASSWORD="
                    + secrets.token_urlsafe(24) + "\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)
    print("Created local .env. Credentials were not printed.")
(root / "deploy/local/backups").mkdir(parents=True, exist_ok=True)
