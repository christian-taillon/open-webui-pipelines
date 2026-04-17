#!/usr/bin/env python3
"""
Generate OpenCode custom-tool wrappers for OpenWebUI tool files.

This script scans a source directory for Python files that expose a top-level
`Tools` class, discovers public tool methods, and emits JavaScript wrappers
plus a shared Python runner under `.opencode/tools/`.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SKIP_PARAMS = {
    "self",
    "__event_emitter__",
    "__user__",
    "__request__",
    "__metadata__",
    "__files__",
    "__context__",
}
SKIP_METHOD_NAMES = {
    "get_headers",
    "escape_tql_string",
    "build_fields_params",
    "build_fields_params_async",
    "detect_ioc_type",
    "query_threatconnect_api",
    "query_threatconnect_api_async",
    "get_available_fields",
    "get_available_fields_list",
}
SKIP_METHOD_PREFIXES = ("build_", "detect_", "escape_", "query_")
RUNNER_NAME = "_owui_runner.py"


@dataclass
class ParameterSpec:
    name: str
    schema_expr: str
    description: str
    required: bool


@dataclass
class MethodSpec:
    name: str
    description: str
    params: list[ParameterSpec]


@dataclass
class ToolFileSpec:
    source_relpath: str
    wrapper_basename: str
    module_title: str
    methods: list[MethodSpec]


def parse_metadata(docstring: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in docstring.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def annotation_to_schema(annotation: ast.AST | None, required: bool) -> str:
    base = "tool.schema.any()"
    if annotation is None:
        base = "tool.schema.any()"
    elif isinstance(annotation, ast.Name):
        mapping = {
            "str": "tool.schema.string()",
            "int": "tool.schema.number().int()",
            "float": "tool.schema.number()",
            "bool": "tool.schema.boolean()",
            "dict": "tool.schema.any()",
            "list": "tool.schema.array(tool.schema.any())",
            "Any": "tool.schema.any()",
        }
        base = mapping.get(annotation.id, "tool.schema.any()")
    elif isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation_to_schema(ast.Name(id=annotation.value), required)
    elif isinstance(annotation, ast.Subscript):
        target = annotation.value
        if isinstance(target, ast.Name) and target.id in {"Optional"}:
            return annotation_to_schema(_slice_value(annotation), False)
        if isinstance(target, ast.Name) and target.id == "list":
            base = "tool.schema.array(tool.schema.any())"
        elif isinstance(target, ast.Name) and target.id == "dict":
            base = "tool.schema.any()"
        elif isinstance(target, ast.Name) and target.id == "Literal":
            literal_values = extract_literal_values(_slice_value(annotation))
            if literal_values:
                joined = ", ".join(json.dumps(value) for value in literal_values)
                base = f"tool.schema.enum([{joined}])"
        elif isinstance(target, ast.Name) and target.id == "Union":
            members = extract_union_members(_slice_value(annotation))
            non_none_members = [node for node in members if not is_none_node(node)]
            if len(non_none_members) == 1 and len(non_none_members) != len(members):
                return annotation_to_schema(non_none_members[0], False)
        elif isinstance(target, ast.Attribute) and target.attr == "Optional":
            return annotation_to_schema(_slice_value(annotation), False)
        else:
            base = "tool.schema.any()"
    elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        members = flatten_bitor(annotation)
        non_none_members = [node for node in members if not is_none_node(node)]
        if len(non_none_members) == 1 and len(non_none_members) != len(members):
            return annotation_to_schema(non_none_members[0], False)
    schema = base
    if not required:
        schema = f"{schema}.optional()"
    return schema


def _slice_value(node: ast.Subscript) -> ast.AST:
    return node.slice


def is_none_node(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and node.value is None
        or isinstance(node, ast.Name)
        and node.id == "None"
    )


def flatten_bitor(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return flatten_bitor(node.left) + flatten_bitor(node.right)
    return [node]


def extract_union_members(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Tuple):
        return list(node.elts)
    return [node]


def extract_literal_values(node: ast.AST) -> list[str]:
    members = extract_union_members(node)
    values: list[str] = []
    for member in members:
        if isinstance(member, ast.Constant) and isinstance(member.value, str):
            values.append(member.value)
        else:
            return []
    return values


def count_defaults(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return len(fn.args.defaults)


def method_parameters(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ParameterSpec] | None:
    positional = list(fn.args.args)
    defaults = list(fn.args.defaults)
    default_offset = len(positional) - len(defaults)
    result: list[ParameterSpec] = []
    for index, arg in enumerate(positional):
        if arg.arg in SKIP_PARAMS:
            continue
        has_default = index >= default_offset
        if arg.arg.startswith("__") and not has_default:
            return None
        required = not has_default
        schema_expr = annotation_to_schema(arg.annotation, required)
        result.append(
            ParameterSpec(
                name=arg.arg,
                schema_expr=schema_expr,
                description=f"{arg.arg} parameter",
                required=required,
            )
        )
    return result


def discover_tool_file(path: Path, root: Path) -> ToolFileSpec | None:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    module_docstring = ast.get_docstring(module) or ""
    metadata = parse_metadata(module_docstring)
    tools_class = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "Tools"
        ),
        None,
    )
    if tools_class is None:
        return None

    methods: list[MethodSpec] = []
    for node in tools_class.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == "__init__" or node.name.startswith("_"):
            continue
        if node.name in SKIP_METHOD_NAMES or node.name.startswith(SKIP_METHOD_PREFIXES):
            continue
        params = method_parameters(node)
        if params is None:
            continue
        description = (ast.get_docstring(node) or "").strip().splitlines()
        first_line = description[0] if description else f"Run `{node.name}` from `{path.name}`."
        methods.append(MethodSpec(name=node.name, description=first_line, params=params))

    if not methods:
        return None

    relpath = path.relative_to(root).as_posix()
    wrapper_basename = f"owui_{path.stem}"
    module_title = metadata.get("title", path.stem)
    return ToolFileSpec(
        source_relpath=relpath,
        wrapper_basename=wrapper_basename,
        module_title=module_title,
        methods=methods,
    )


def js_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def render_wrapper(spec: ToolFileSpec) -> str:
    lines = [
        'import { tool } from "@opencode-ai/plugin"',
        'import path from "path"',
        "",
        f"const RUNNER = {js_string(f'.opencode/tools/{RUNNER_NAME}')}",
        f"const SOURCE = {js_string(spec.source_relpath)}",
        "",
    ]
    for method in spec.methods:
        export_name = method.name
        lines.append(f"export const {export_name} = tool({{")
        lines.append(
            f"  description: {js_string(f'{method.description} [OpenWebUI: {spec.module_title}]')},"
        )
        lines.append("  args: {")
        for param in method.params:
            lines.append(
                f"    {param.name}: {param.schema_expr}.describe({js_string(param.description)}),"
            )
        lines.append("  },")
        lines.append("  async execute(args, context) {")
        lines.append("    const runner = path.join(context.worktree, RUNNER)")
        lines.append("    const source = path.join(context.worktree, SOURCE)")
        lines.append(
            f"    const result = await Bun.$`python3 ${{runner}} ${{source}} {method.name} ${{JSON.stringify(args)}}`.text()"
        )
        lines.append("    return result.trim()")
        lines.append("  },")
        lines.append("})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_manifest(specs: Iterable[ToolFileSpec]) -> str:
    manifest = []
    for spec in specs:
        manifest.append(
            {
                "source": spec.source_relpath,
                "wrapper": f"{spec.wrapper_basename}.js",
                "tools": [method.name for method in spec.methods],
            }
        )
    return json.dumps({"generated": manifest}, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="tools")
    parser.add_argument("--output-dir", default=".opencode/tools")
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    root = Path.cwd()
    source_dir = (root / args.source_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    only = {item for item in args.only}
    specs: list[ToolFileSpec] = []
    for path in sorted(source_dir.glob("*.py")):
        if only and path.name not in only and path.stem not in only:
            continue
        spec = discover_tool_file(path, root)
        if spec:
            specs.append(spec)

    for spec in specs:
        wrapper_path = output_dir / f"{spec.wrapper_basename}.js"
        wrapper_path.write_text(render_wrapper(spec), encoding="utf-8")

    manifest_path = output_dir / "_owui_manifest.json"
    manifest_path.write_text(render_manifest(specs), encoding="utf-8")
    print(f"Generated {len(specs)} OpenCode wrapper file(s) in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
