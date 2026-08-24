"""Tokenization for a small POSIX-style shell grammar."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from .errors import LexerError


PLAIN = "plain"
SINGLE = "single"
DOUBLE = "double"


@dataclass(frozen=True)
class Part:
    text: str
    mode: str = PLAIN


@dataclass(frozen=True)
class Word:
    parts: tuple[Part, ...]

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts)

    @property
    def has_quoted_part(self) -> bool:
        return any(part.mode != PLAIN for part in self.parts)

    def split_assignment(self) -> tuple[str, "Word"] | None:
        text = self.text
        if "=" not in text:
            return None
        name, _sep, _value = text.partition("=")
        if not is_name(name):
            return None
        return name, self.slice_from(len(name) + 1)

    def slice_from(self, offset: int) -> "Word":
        parts: list[Part] = []
        cursor = 0
        for part in self.parts:
            next_cursor = cursor + len(part.text)
            if next_cursor > offset:
                start = max(0, offset - cursor)
                parts.append(Part(part.text[start:], part.mode))
            cursor = next_cursor
        return Word(tuple(parts))


@dataclass(frozen=True)
class Operator:
    value: str


Token = Word | Operator


def is_name(value: str) -> bool:
    if not value:
        return False
    if not (value[0].isalpha() or value[0] == "_"):
        return False
    return all(char.isalnum() or char == "_" for char in value[1:])


def lex(source: str) -> list[Token]:
    tokens: list[Token] = []
    parts: list[Part] = []
    i = 0

    def add_part(text: str, mode: str = PLAIN, keep_empty: bool = False) -> None:
        if not text and not keep_empty:
            return
        if parts and parts[-1].mode == mode and text:
            parts[-1] = Part(parts[-1].text + text, mode)
            return
        parts.append(Part(text, mode))

    def flush_word() -> None:
        nonlocal parts
        if parts:
            tokens.append(Word(tuple(parts)))
            parts = []

    def last_token_allows_linebreak() -> bool:
        return bool(
            tokens
            and isinstance(tokens[-1], Operator)
            and tokens[-1].value in {";", "&&", "||", "|"}
        )

    def current_word_text() -> str:
        return "".join(part.text for part in parts)

    while i < len(source):
        char = source[i]

        if char == "\n":
            flush_word()
            if not last_token_allows_linebreak():
                tokens.append(Operator(";"))
            i += 1
            continue

        if char.isspace():
            flush_word()
            i += 1
            continue

        if char == "#" and not parts:
            while i < len(source) and source[i] != "\n":
                i += 1
            continue

        if not parts and char.isdigit():
            j = i
            while j < len(source) and source[j].isdigit():
                j += 1
            if j < len(source) and source[j] in "<>":
                op_end = j + 1
                if op_end < len(source) and source[j : op_end + 1] in {">>", ">&", ">|", "<&", "<<", "<<-", "<>"}:
                    op_end += 1
                tokens.append(Operator(source[i:op_end]))
                i = op_end
                continue

        three = source[i : i + 3]
        if three == "<<-":
            flush_word()
            tokens.append(Operator(three))
            i += 3
            continue

        two = source[i : i + 2]
        if two in {"&&", "||", ">>", ">&", ">|", "<&", "<<", "<>", ";;"}:
            flush_word()
            tokens.append(Operator(two))
            i += 2
            continue

        if char in ";|&<>()":
            flush_word()
            tokens.append(Operator(char))
            i += 1
            continue

        if char == "'":
            i += 1
            start = i
            while i < len(source) and source[i] != "'":
                i += 1
            if i >= len(source):
                raise LexerError("unterminated single quote")
            add_part(source[start:i], SINGLE, keep_empty=True)
            i += 1
            continue

        if char == '"':
            i += 1
            chunk: list[str] = []
            def flush_double_chunk() -> None:
                if chunk:
                    add_part("".join(chunk), DOUBLE)
                    chunk.clear()

            while i < len(source):
                char = source[i]
                if char == '"':
                    break
                if char == "$" and source.startswith("$((", i):
                    flush_double_chunk()
                    value, i = read_arithmetic(source, i)
                    add_part(value, DOUBLE)
                    continue
                if char == "$" and i + 1 < len(source) and source[i + 1] == "(":
                    flush_double_chunk()
                    value, i = read_balanced(source, i, "$(", ")")
                    add_part(value, DOUBLE)
                    continue
                if char == "$" and i + 1 < len(source) and source[i + 1] == "{":
                    flush_double_chunk()
                    value, i = read_balanced(source, i, "${", "}")
                    add_part(value, DOUBLE)
                    continue
                if char == "`":
                    flush_double_chunk()
                    value, i = read_backtick(source, i)
                    add_part(value, DOUBLE)
                    continue
                if char == "\\":
                    if i + 1 >= len(source):
                        chunk.append("\\")
                        i += 1
                        continue
                    nxt = source[i + 1]
                    if nxt == "\n":
                        i += 2
                        continue
                    if nxt in {'$', "`", '"', "\\"}:
                        flush_double_chunk()
                        add_part(nxt, SINGLE, keep_empty=True)
                    else:
                        chunk.append("\\")
                        chunk.append(nxt)
                    i += 2
                    continue
                chunk.append(char)
                i += 1
            if i >= len(source):
                raise LexerError("unterminated double quote")
            if chunk:
                flush_double_chunk()
            else:
                add_part("", DOUBLE, keep_empty=True)
            i += 1
            continue

        if char == "\\":
            if should_keep_windows_path_separator(source, i, current_word_text()):
                add_part("\\", PLAIN)
                i += 1
                continue
            if i + 1 < len(source) and source[i + 1] == "\n":
                i += 2
                continue
            if i + 1 < len(source):
                add_part(source[i + 1], PLAIN)
                i += 2
            else:
                add_part("\\", PLAIN)
                i += 1
            continue

        if char == "$" and source.startswith("$((", i):
            value, i = read_arithmetic(source, i)
            add_part(value, PLAIN)
            continue

        if char == "$" and i + 1 < len(source) and source[i + 1] == "(":
            value, i = read_balanced(source, i, "$(", ")")
            add_part(value, PLAIN)
            continue

        if char == "$" and i + 1 < len(source) and source[i + 1] == "{":
            value, i = read_balanced(source, i, "${", "}")
            add_part(value, PLAIN)
            continue

        if char == "`":
            value, i = read_backtick(source, i)
            add_part(value, PLAIN)
            continue

        add_part(char, PLAIN)
        i += 1

    flush_word()
    while tokens and isinstance(tokens[-1], Operator) and tokens[-1].value == ";":
        tokens.pop()
    return tokens


def should_keep_windows_path_separator(source: str, index: int, word_text: str) -> bool:
    if os.name != "nt" or index + 1 >= len(source):
        return False
    next_char = source[index + 1]
    if next_char.isspace() or next_char in ";|&<>()":
        return False
    if word_text in {".", ".."}:
        return True
    if len(word_text) == 2 and word_text[0].isalpha() and word_text[1] == ":":
        return True
    return "\\" in word_text


def dump_tokens(tokens: Iterable[Token]) -> list[str]:
    result: list[str] = []
    for token in tokens:
        if isinstance(token, Operator):
            result.append(f"op:{token.value}")
        else:
            result.append(f"word:{token.text}")
    return result


def read_balanced(source: str, start: int, opener: str, closer: str) -> tuple[str, int]:
    i = start + len(opener)
    depth = 1
    quote: str | None = None
    escaped = False
    while i < len(source):
        char = source[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if char == "\\":
            escaped = True
            i += 1
            continue
        if quote:
            if char == quote:
                quote = None
            i += 1
            continue
        if char in {"'", '"'}:
            quote = char
            i += 1
            continue
        if opener == "$(" and source.startswith("$(", i):
            depth += 1
            i += 2
            continue
        if opener == "${" and source.startswith("${", i):
            depth += 1
            i += 2
            continue
        if char == closer:
            depth -= 1
            i += 1
            if depth == 0:
                return source[start:i], i
            continue
        i += 1
    raise LexerError("unterminated command substitution" if opener == "$(" else "unterminated parameter expansion")


def read_backtick(source: str, start: int) -> tuple[str, int]:
    i = start + 1
    escaped = False
    while i < len(source):
        char = source[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if char == "\\":
            escaped = True
            i += 1
            continue
        if char == "`":
            return source[start : i + 1], i + 1
        i += 1
    raise LexerError("unterminated command substitution")


def read_arithmetic(source: str, start: int) -> tuple[str, int]:
    i = start + 3
    depth = 1
    quote: str | None = None
    escaped = False
    while i < len(source):
        char = source[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if char == "\\":
            escaped = True
            i += 1
            continue
        if quote:
            if char == quote:
                quote = None
            i += 1
            continue
        if char in {"'", '"'}:
            quote = char
            i += 1
            continue
        if source.startswith("((", i):
            depth += 1
            i += 2
            continue
        if source.startswith("))", i):
            depth -= 1
            i += 2
            if depth == 0:
                return source[start:i], i
            continue
        i += 1
    raise LexerError("unterminated arithmetic expansion")
