"""Parse Telegram HTML exports and recover message timestamps."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path


MESSAGE_ID_RE = re.compile(r"^message(-?\d+)$")
TITLE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2}):(\d{2}) UTC([+-]\d{2}):?(\d{2})")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class HtmlMessage:
    message_id: int
    group_title: str
    datetime_utc: datetime
    text: str


def normalize_text(value: str) -> str:
    return SPACE_RE.sub(" ", html.unescape(value or "").strip()).casefold()


def parse_telegram_title(value: str) -> datetime | None:
    match = TITLE_RE.match(value.strip())
    if not match:
        return None
    day, month, year, hour, minute, second, offset_hour, offset_minute = match.groups()
    sign = 1 if offset_hour.startswith("+") else -1
    offset = timezone(timedelta(minutes=sign * (abs(int(offset_hour)) * 60 + int(offset_minute))))
    local_datetime = datetime(
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
        int(second),
        tzinfo=offset,
    )
    return local_datetime.astimezone(timezone.utc)


class TelegramHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.group_title = ""
        self.messages: list[HtmlMessage] = []
        self._in_header_title = False
        self._header_parts: list[str] = []
        self._current_id: int | None = None
        self._current_datetime: datetime | None = None
        self._in_date = False
        self._in_text = False
        self._text_depth = 0
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "div" and {"text", "bold"}.issubset(classes) and not self.group_title:
            self._in_header_title = True
            self._header_parts = []
            return
        if tag == "div" and "message" in classes:
            match = MESSAGE_ID_RE.match(attr.get("id", ""))
            if match:
                self._current_id = int(match.group(1))
                self._current_datetime = None
                self._text_parts = []
            return
        if self._current_id is not None and tag == "div" and {"date", "details"}.issubset(classes):
            self._current_datetime = parse_telegram_title(attr.get("title", ""))
            self._in_date = True
            return
        if self._current_id is not None and tag == "div" and "text" in classes:
            self._in_text = True
            self._text_depth = 1
            self._text_parts = []
            return
        if self._in_text:
            self._text_depth += 1
            if tag in {"br", "p", "div"}:
                self._text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._in_header_title and tag == "div":
            self.group_title = normalize_text(" ".join(self._header_parts))
            self._in_header_title = False
            return
        if self._in_date and tag == "div":
            self._in_date = False
            return
        if self._in_text:
            if tag in {"p", "div"}:
                self._text_parts.append(" ")
            self._text_depth -= 1
            if self._text_depth <= 0:
                self._in_text = False
                self._flush_message()
                return

    def handle_data(self, data: str) -> None:
        if self._in_header_title:
            self._header_parts.append(data)
        if self._in_text:
            self._text_parts.append(data)

    def _flush_message(self) -> None:
        if self._current_id is None or self._current_datetime is None:
            return
        text = normalize_text(" ".join(self._text_parts))
        if not text:
            return
        self.messages.append(
            HtmlMessage(
                message_id=self._current_id,
                group_title=self.group_title,
                datetime_utc=self._current_datetime,
                text=text,
            )
        )


def parse_telegram_html_file(path: Path) -> list[HtmlMessage]:
    parser = TelegramHtmlParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.messages


def parse_telegram_html_dir(directory: Path) -> list[HtmlMessage]:
    messages: list[HtmlMessage] = []
    for path in sorted(directory.glob("messages*.html")):
        messages.extend(parse_telegram_html_file(path))
    return messages
