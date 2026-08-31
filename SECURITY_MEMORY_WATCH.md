# Lio v3 — Memory, Security & Watch Center Foundation

## تمت إضافته
- User profile storage.
- Project storage.
- Persistent conversation history.
- Long-term memory table.
- Approval queue.
- Audit log.
- Website Watchlist database and API.
- Minimum watch frequency guardrail: 60 minutes.

## مستويات التنفيذ
### Automatic
Research, reading, analysis, translation, monitoring specifications.

### Approval required
Sending messages, publishing, bookings, important edits.

### Explicit confirmation
Payments, purchases, transfers, signatures, contracts and other commitments.

## Production migration
SQLite is intentionally only the local development database.
Before production deployment, move durable data to managed PostgreSQL,
enable authenticated user IDs, encrypted secrets, backups and row-level access controls.

## Watch Center next step
A background worker will periodically fetch enabled watch items, compare fingerprints,
use Lio to judge whether the change matches the user's rule, then create an alert.
