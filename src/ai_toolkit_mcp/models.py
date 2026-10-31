"""Model presets read from the ai-toolkit UI's model definitions (GET /api/model_archs).

The UI ships each model's training defaults as transpiled JS source, e.g.
    "config.process[0].model.qtype": ["convrot8", "qfloat8"]   // [selected, unselected]
This module pulls those literal values out so presets track whatever ai-toolkit version is installed.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any


class _Unknown:
    """A JS expression we can't evaluate (identifier, function call)."""


UNKNOWN = _Unknown()
UNDEFINED = object()

_NUMBER = re.compile(r"-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
_IDENT = re.compile(r"[A-Za-z_$][\w$.]*")


class _Parser:
    def __init__(self, text: str, pos: int = 0):
        self.s, self.i = text, pos

    def ws(self) -> None:
        while self.i < len(self.s):
            if self.s[self.i].isspace():
                self.i += 1
            elif self.s.startswith("//", self.i):
                end = self.s.find("\n", self.i)
                self.i = len(self.s) if end < 0 else end
            elif self.s.startswith("/*", self.i):
                end = self.s.find("*/", self.i)
                self.i = len(self.s) if end < 0 else end + 2
            else:
                return

    def peek(self) -> str:
        self.ws()
        return self.s[self.i] if self.i < len(self.s) else ""

    def value(self) -> Any:
        ch = self.peek()
        if ch in "\"'":
            return self.string()
        if ch == "[":
            return self.array()
        if ch == "{":
            return self.obj()
        m = _NUMBER.match(self.s, self.i)
        if m:
            self.i = m.end()
            return float(m.group()) if any(c in m.group() for c in ".eE") else int(m.group())
        m = _IDENT.match(self.s, self.i)
        if m:
            self.i = m.end()
            word = m.group()
            if self.peek() == "(":
                self.skip_balanced("(", ")")
                return UNKNOWN
            return {"true": True, "false": False, "null": None, "undefined": UNDEFINED}.get(word, UNKNOWN)
        raise ValueError(f"unexpected {ch!r} at {self.i}")

    def string(self) -> str:
        quote, start = self.s[self.i], self.i
        self.i += 1
        while self.s[self.i] != quote:
            self.i += 2 if self.s[self.i] == "\\" else 1
        self.i += 1
        body = self.s[start + 1 : self.i - 1]
        if quote == "'":
            body = body.replace('\\"', '"').replace("\\'", "'").replace('"', '\\"')
        return json.loads(f'"{body}"')

    def array(self) -> list:
        self.i += 1
        out = []
        while self.peek() != "]":
            out.append(self.value())
            if self.peek() == ",":
                self.i += 1
        self.i += 1
        return out

    def obj(self) -> dict:
        self.i += 1
        out: dict = {}
        while self.peek() != "}":
            if self.peek() in "\"'":
                key = self.string()
            else:
                m = _IDENT.match(self.s, self.i)
                if not m:
                    raise ValueError(f"bad key at {self.i}")
                key, self.i = m.group(), m.end()
            if self.peek() == ":":
                self.i += 1
                out[key] = self.value()
            if self.peek() == ",":
                self.i += 1
        self.i += 1
        return out

    def skip_balanced(self, open_ch: str, close_ch: str) -> None:
        depth = 0
        while self.i < len(self.s):
            ch = self.s[self.i]
            if ch in "\"'":
                self.string()
                continue
            depth += ch == open_ch
            depth -= ch == close_ch
            self.i += 1
            if depth == 0:
                return


@dataclass
class ModelArch:
    name: str
    label: str
    group: str | None = None
    is_video: bool = False
    gate_url: str | None = None
    defaults: dict[str, list] = field(default_factory=dict)
    additional_sections: list[str] = field(default_factory=list)


_ENTRY = re.compile(r'\bname:\s*"([^"]+)",\s*label:\s*"([^"]*)"')


def _field(chunk: str, key: str) -> Any:
    m = re.search(rf"\b{key}:\s*", chunk)
    if not m:
        return None
    try:
        return _Parser(chunk, m.end()).value()
    except (ValueError, IndexError):
        return None


def parse_model_archs(payload: dict) -> dict[str, ModelArch]:
    archs: dict[str, ModelArch] = {}
    for module in payload.get("modules", []):
        code = module.get("code", "")
        starts = list(_ENTRY.finditer(code))
        for n, m in enumerate(starts):
            chunk = code[m.start() : starts[n + 1].start() if n + 1 < len(starts) else len(code)]
            defaults = _field(chunk, "defaults")
            sections = _field(chunk, "additionalSections")
            group, gate = _field(chunk, "group"), _field(chunk, "gateUrl")
            archs[m.group(1)] = ModelArch(
                name=m.group(1),
                label=m.group(2),
                group=group if isinstance(group, str) else None,
                is_video=_field(chunk, "isVideoModel") is True,
                gate_url=gate if isinstance(gate, str) else None,
                defaults={k: v for k, v in (defaults or {}).items() if isinstance(v, list) and len(v) == 2},
                additional_sections=[s for s in (sections or []) if isinstance(s, str)],
            )
    return archs


def _assign(proc: dict, key: str, value: Any) -> None:
    """Set a `config.process[0].a.b` / `...datasets[x].c` key on process[0]; UNDEFINED deletes it."""
    path = key.removeprefix("config.process[0].")
    if path.startswith("datasets[x]."):
        targets, leaf_path = proc.get("datasets", []), path[len("datasets[x].") :]
    else:
        targets, leaf_path = [proc], path
    for target in targets:
        *parents, leaf = leaf_path.split(".")
        node = target
        for p in parents:
            node = node.setdefault(p, {})
        if value is UNDEFINED:
            node.pop(leaf, None)
        else:
            node[leaf] = value


def apply_model(proc: dict, archs: dict[str, ModelArch], new_name: str) -> None:
    """Switch process[0] to a model the way the UI's model dropdown does (simplified)."""
    new = archs[new_name]
    current = archs.get((proc.get("model") or {}).get("arch", ""))
    if current and current.name != new_name:
        for key, (_, unselected) in current.defaults.items():
            if unselected is not UNKNOWN:
                _assign(proc, key, unselected)
    proc.setdefault("model", {})["arch"] = new_name
    if "model.low_vram" not in new.additional_sections:
        proc["model"]["low_vram"] = False
    for key, (selected, _) in new.defaults.items():
        if selected is not UNKNOWN:
            _assign(proc, key, selected)
