#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sobrescreve o campo 'id' de cada elemento em arquivos JSON."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="sentinel_replica_jsons",
        help="Diretorio com os arquivos JSON a serem atualizados.",
    )
    return parser.parse_args()


def update_file(json_path: Path) -> int:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{json_path} nao contem uma lista JSON na raiz.")

    for item in tqdm(data, desc=json_path.name, unit="item", leave=False):
        if not isinstance(item, dict):
            raise ValueError(f"{json_path} contem item que nao e objeto JSON.")
        item["id"] = str(uuid4())

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return len(data)


def main() -> None:
    args = parse_args()
    target_dir = Path(args.directory)

    if not target_dir.is_dir():
        raise FileNotFoundError(f"Diretorio nao encontrado: {target_dir}")

    json_files = sorted(target_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"Nenhum arquivo JSON encontrado em: {target_dir}")

    total_items = 0
    for json_file in tqdm(json_files, desc="Arquivos", unit="arquivo"):
        total_items += update_file(json_file)

    print(f"IDs sobrescritos em {len(json_files)} arquivos e {total_items} itens.")


if __name__ == "__main__":
    main()
