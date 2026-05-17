import argparse
import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError, UsernameInvalidError, UsernameNotOccupiedError


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "sentinel_replica_jsons"
DEFAULT_SESSION_PATH = ROOT_DIR / "session_sentinel"
DEFAULT_START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
# Data de corte: alinhada com o dado mais recente dos grupos existentes (2026-05-09)
DEFAULT_END_DATE = datetime(2026, 5, 10, tzinfo=timezone.utc)  # exclusive upper bound (day + 1)

GROUPS = [
    "joinhackingarmy",
    "HackingBlogsGroup",
    "cloudandcybersecurity",
    "cybdetective",
    "cissp",
    "PHOfficial",
    "WokeIntelDrops",
    "itsectalk",
    "hackers_asylum",
    "cybersecurityexperts",
    "vxunderground",
    "RedPacketSecurity",
    "twittercvenews",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coleta mensagens recentes dos grupos do Telegram e faz merge "
            "incremental com os JSONs já salvos."
        )
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=GROUPS,
        help="Lista de grupos públicos a coletar.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Diretório onde os JSONs serão salvos.",
    )
    parser.add_argument(
        "--session-path",
        default=str(DEFAULT_SESSION_PATH),
        help="Caminho base da sessão do Telethon.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=None,
        help=(
            "Data mínima manual no formato YYYY-MM-DD. "
            "Se omitida, usa o último dia já salvo no arquivo."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=parse_end_date,
        default=None,
        help=(
            "Data final inclusiva no formato YYYY-MM-DD. "
            "Se omitida, usa DEFAULT_END_DATE (2026-05-09) para alinhar com os dados existentes."
        ),
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=2,
        help=(
            "Dias de sobreposição na retomada automática. "
            "Útil para evitar perdas na coleta incremental."
        ),
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help=(
            "Ignora o JSON existente do grupo e recoleta desde --start-date "
            "ou 2023-01-01, preservando timestamps completos."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Não cria backup .bak antes de substituir um JSON existente.",
    )
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="Permite que --full-refresh substitua um JSON existente por uma coleta menor.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite opcional de mensagens por grupo, útil para teste.",
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def parse_end_date(value: str) -> datetime:
    return parse_date(value) + timedelta(days=1)


def parse_saved_datetime(item: dict) -> datetime | None:
    datetime_value = item.get("datetime_utc")
    if isinstance(datetime_value, str) and datetime_value:
        try:
            return datetime.fromisoformat(datetime_value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            pass

    date_value = item.get("date")
    if isinstance(date_value, str) and date_value:
        return parse_date(date_value)
    return None


def load_credentials() -> tuple[int, str]:
    load_dotenv(ROOT_DIR / ".env")

    api_id_raw = os.getenv("API_ID")
    api_hash = os.getenv("API_HASH")

    if not api_id_raw or not api_hash:
        raise RuntimeError(
            "API_ID e API_HASH precisam estar definidos no arquivo .env da raiz do projeto."
        )

    return int(api_id_raw), api_hash


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_existing_messages(filepath: Path, full_refresh: bool = False) -> tuple[list[dict], set[int], set[tuple[str, str]], datetime | None]:
    if full_refresh or not filepath.exists():
        return [], set(), set(), None

    with filepath.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{filepath} não contém uma lista JSON válida.")

    existing_ids: set[int] = set()
    existing_signatures: set[tuple[str, str]] = set()
    latest_date: datetime | None = None

    for item in data:
        if not isinstance(item, dict):
            continue

        message_id = item.get("_id")
        if isinstance(message_id, int):
            existing_ids.add(message_id)

        saved_datetime = parse_saved_datetime(item)
        date_value = saved_datetime.isoformat() if saved_datetime is not None else item.get("date")
        message_text = item.get("message")
        if isinstance(date_value, str) and isinstance(message_text, str):
            existing_signatures.add((date_value, message_text))
            if saved_datetime is not None and (latest_date is None or saved_datetime > latest_date):
                latest_date = saved_datetime

    return data, existing_ids, existing_signatures, latest_date


def build_entry(message) -> dict:
    date_utc = message.date.astimezone(timezone.utc)
    return {
        "_id": message.id,
        "date": date_utc.strftime("%Y-%m-%d"),
        "datetime_utc": date_utc.isoformat(),
        "timestamp": int(date_utc.timestamp()),
        "message": message.text,
        "id": str(uuid.uuid4()),
    }


def save_messages(filepath: Path, messages: list[dict]) -> None:
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=2)
    tmp_path.replace(filepath)


def backup_existing_file(filepath: Path) -> Path | None:
    if not filepath.exists():
        return None
    backup_path = filepath.with_suffix(filepath.suffix + ".bak")
    shutil.copy2(filepath, backup_path)
    return backup_path


def count_existing_file(filepath: Path) -> int:
    if not filepath.exists():
        return 0
    with filepath.open(encoding="utf-8") as file:
        payload = json.load(file)
    return len(payload) if isinstance(payload, list) else 0


def resolve_start_date(
    manual_start_date: datetime | None,
    latest_saved_date: datetime | None,
    overlap_days: int,
) -> datetime:
    if manual_start_date is not None:
        return manual_start_date

    if latest_saved_date is None:
        return DEFAULT_START_DATE

    overlap = timedelta(days=max(overlap_days, 0))
    return max(DEFAULT_START_DATE, latest_saved_date - overlap)


async def collect_group(
    client: TelegramClient,
    group: str,
    output_dir: Path,
    manual_start_date: datetime | None,
    manual_end_date: datetime | None,
    overlap_days: int,
    limit: int | None,
    full_refresh: bool,
    no_backup: bool,
    allow_shrink: bool,
) -> None:
    filepath = output_dir / f"{group}.json"
    old_count = count_existing_file(filepath)
    existing_messages, existing_ids, existing_signatures, latest_saved_date = load_existing_messages(filepath, full_refresh)

    start_date = resolve_start_date(manual_start_date, latest_saved_date, overlap_days)
    end_date = manual_end_date or DEFAULT_END_DATE

    latest_label = latest_saved_date.isoformat() if latest_saved_date else "arquivo novo"
    print(
        f"{group}: retomando de {start_date.isoformat()} "
        f"(último salvo: {latest_label}) até {end_date.isoformat()}"
    )

    new_messages: list[dict] = []
    reached_end = False

    while True:
        try:
            async for message in client.iter_messages(group, offset_date=start_date, reverse=True, limit=limit):
                message_date = message.date.astimezone(timezone.utc)

                if message_date < start_date:
                    continue
                if message_date >= end_date:
                    reached_end = True
                    break

                if not message.text:
                    continue

                if message.id in existing_ids:
                    continue

                signature = (message_date.isoformat(), message.text)
                if signature in existing_signatures:
                    continue

                entry = build_entry(message)
                new_messages.append(entry)
                existing_ids.add(message.id)
                existing_signatures.add(signature)

                if len(new_messages) % 500 == 0:
                    print(f"  {group}: {len(new_messages)} mensagens novas encontradas...")

            break

        except (UsernameInvalidError, UsernameNotOccupiedError) as error:
            print(f"  -> {group}: username inválido ou inexistente, pulando ({error})")
            return

        except FloodWaitError as error:
            wait_seconds = error.seconds + 5
            print(f"  {group}: FloodWait de {error.seconds}s, aguardando {wait_seconds}s...")
            await asyncio.sleep(wait_seconds)

    if new_messages:
        output_path = filepath
        if full_refresh and old_count and len(new_messages) < old_count and not allow_shrink:
            output_path = filepath.with_suffix(filepath.suffix + ".refresh.json")
            print(
                f"  {group}: coleta nova menor que a existente "
                f"({len(new_messages)} < {old_count}); salvando em {output_path} "
                "sem substituir. Use --allow-shrink para forçar."
            )
        elif full_refresh and old_count and not no_backup:
            backup_path = backup_existing_file(filepath)
            if backup_path is not None:
                print(f"  {group}: backup salvo em {backup_path}")

        merged_messages = new_messages + existing_messages
        save_messages(output_path, merged_messages)
        status = "end-date alcançado" if reached_end else "fim do histórico/limite alcançado"
        print(f"  -> {group}: {len(new_messages)} novas, total {len(merged_messages)} ({status})")
        return

    print(f"  -> {group}: nenhuma mensagem nova")


async def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    api_id, api_hash = load_credentials()
    client = TelegramClient(str(Path(args.session_path)), api_id, api_hash)

    await client.start()
    try:
        for group in args.groups:
            await collect_group(
                client=client,
                group=group,
                output_dir=output_dir,
                manual_start_date=args.start_date,
                manual_end_date=args.end_date,
                overlap_days=args.overlap_days,
                limit=args.limit,
                full_refresh=args.full_refresh,
                no_backup=args.no_backup,
                allow_shrink=args.allow_shrink,
            )
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
