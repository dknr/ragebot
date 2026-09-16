# ragebot

A Matrix bot client built on [mautrix-go](https://maunium.net/go/mautrix) that:

- Supports **E2EE** (Olm/Megolm via the mautrix crypto machine) and **device verification**
  (auto-accepts incoming SAS verification requests).
- **Auto-joins** rooms when invited.
- **Logs every received event** (messages, invites, state, to-device, everything) with zerolog.
- On startup: **logs in**, **verifies its own device** if necessary (with a recovery key or by
  generating new cross-signing keys), and **signs out any other sessions**.

## Building

mautrix-go ships two olm implementations: the C `libolm` (needs the olm library) and the pure Go
`goolm`. The build uses the `goolm` tag to avoid the cgo dependency:

```sh
make build   # or: go build -tags goolm -o ragebot .
```

## Configuration

Create `config.json` (or pass `-config <path>`):

```json
{
  "homeserver": "https://matrix.example.org",
  "user": "@bot:example.org",
  "password": "change-me",
  "recovery_key": "",
  "database": "ragebot.db",
  "pickle_key": "ragebot"
}
```

- `homeserver`, `user`, `password` are required.
- `recovery_key` is optional. It is only needed when the bot's own device is not verified but
  cross-signing keys already exist on the server (from a previous run). If it is empty and no
  keys exist, the bot generates new cross-signing keys and **logs the new recovery key** on
  startup — save it into the config for future runs.
- `database` is the SQLite file for the crypto/state stores.
- `pickle_key` encrypts the keys stored in the database.

## Running

```sh
./ragebot -config config.json
```
