"""Generate scikit-learn-style class references from live signatures/docstrings."""

import argparse
import html
import importlib
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def signature_block(obj, name, *, is_class=False):
    parameters = []
    keyword_separator = False
    for param in inspect.signature(obj).parameters.values():
        if param.name == "self":
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_separator:
            parameters.append("*")
            keyword_separator = True
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            keyword_separator = True
        parameters.append(str(param.replace(annotation=inspect.Parameter.empty)))
    label = "class " if is_class else ""
    prefix, _, short_name = name.rpartition(".")
    prefix = prefix + "." if prefix else ""
    name = short_name or name
    rendered_name = (
        f'<span class="api-keyword">{label}</span>'
        f'<span class="api-module">{prefix}</span>'
        f'<span class="api-name">{html.escape(name)}</span>'
    )
    if is_class or len(parameters) > 2:
        arguments = "\n".join("    " + html.escape(p) + "," for p in parameters)
        call = f"{rendered_name}(\n{arguments}\n)"
    else:
        call = f"{rendered_name}({html.escape(', '.join(parameters))})"
    source_path = Path(inspect.getsourcefile(obj)).resolve()
    package_root = ROOT / "src"
    if source_path.is_relative_to(package_root):
        line = inspect.getsourcelines(obj)[1]
        source_url = (
            "https://github.com/JackGreenaway/Rgram/blob/main/"
            f"{source_path.relative_to(ROOT).as_posix()}#L{line}"
        )
        source_label = "[source]"
    else:
        relative = source_path.as_posix().split("/sklearn/", 1)[1]
        line = inspect.getsourcelines(obj)[1]
        source_url = (
            "https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/"
            f"{relative}#L{line}"
        )
        source_label = "[inherited API]"
    return (
        '<div class="api-signature">'
        f'<a class="api-source" href="{source_url}">{source_label}</a>'
        f"<pre><code>{call}</code></pre></div>\n"
    )


def normalize_rst(text):
    """Translate inherited scikit-learn markup into ordinary Markdown."""
    text = re.sub(r":math:`(.*?)`", lambda m: "$" + m[1] + "$", text, flags=re.S)
    text = re.sub(r":(?:class|func|meth|mod|attr):`~?([^`]+)`", r"`\1`", text)

    def reference_link(match):
        label, separator, target = match[1].partition(" <")
        if separator:
            target = target.rstrip(">")
        else:
            target = label
        if target == "metadata_routing":
            return "[metadata routing guide](https://scikit-learn.org/stable/metadata_routing.html)"
        return f"`{label}`"

    text = re.sub(r":ref:`([^`]+)`", reference_link, text)
    text = re.sub(r"(?m)^\s*\.\. versionadded:: (.*)$", r"Added in version \1.", text)
    text = re.sub(r"(?m)^\s*\.\. note::\s*$", "**Note:**", text)
    return text.replace("``", "`")


def inline(text):
    text = html.escape(normalize_rst(text))
    text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
    return re.sub(r"\$(.*?)\$", r'<span class="arithmatex">\(\1\)</span>', text)


def render_doc(doc, level, owner=None):
    sections = re.split(
        r"\n([A-Za-z ]+)\n-{3,}\n", inspect.cleandoc(normalize_rst(doc))
    )
    output = [sections[0].replace("``", "`").strip() + "\n"]
    for heading, body in zip(sections[1::2], sections[2::2]):
        if heading == "Examples":
            output.append(f"{'#' * level} Examples\n")
            output.append("```pycon\n" + body.strip() + "\n```\n")
            continue
        if heading == "See Also":
            output.append(f"{'#' * level} See Also\n")
            for entry in re.split(r"\n(?=\S)", body.strip()):
                target, _, description = entry.partition("\n")
                target, _, inline_description = target.partition(" : ")
                description = " ".join((inline_description + " " + description).split())
                links = {
                    "Regressogram": "regressogram.md#rgram.Regressogram",
                    "KernelSmoother": "kernel_smoother.md#rgram.KernelSmoother",
                }
                link = links.get(target.strip())
                if link is None and owner and hasattr(resolve(owner), target.strip()):
                    link = f"#{owner}.{target.strip()}"
                label = f"[{target.strip()}]({link})" if link else f"`{target.strip()}`"
                output.append(f"- {label}: {description}")
            output.append("")
            continue
        output.append(f"{'#' * level} {heading}\n")
        if heading in {"Parameters", "Attributes", "Returns", "Raises"}:
            output.append('<dl class="api-fields">')
            entries = re.split(r"\n(?=\S)", body.strip())
            for entry in entries:
                lines = entry.splitlines()
                term = lines[0]
                description = " ".join(line.strip() for line in lines[1:])
                field_name, separator, field_type = term.partition(" : ")
                label = f'<code class="api-field-name">{inline(field_name)}</code>'
                if separator:
                    label += (
                        f' <span class="api-field-type">{inline(field_type)}</span>'
                    )
                output.append(f"<dt>{label}</dt><dd><p>{inline(description)}</p></dd>")
            output.append("</dl>\n")
        else:
            output.append(body.replace("``", "`").strip() + "\n")
    return "\n".join(output)


def resolve(name):
    module, _, attribute = name.rpartition(".")
    return getattr(importlib.import_module(module), attribute)


def reference(name):
    obj = resolve(name)
    is_class = inspect.isclass(obj)
    output = [
        f'<span id="{name}"></span>',
        f"## {name}",
        signature_block(obj, name, is_class=is_class)
        if not (is_class and issubclass(obj, Warning))
        else "",
        '<div class="api-class-body" markdown="1">',
        render_doc(inspect.getdoc(obj) or "", 3, owner=name),
    ]
    if is_class and not issubclass(obj, Warning):
        methods = [
            (key, value)
            for key, value in inspect.getmembers(obj)
            if callable(value) and (not key.startswith("_") or key == "__call__")
        ]
        table = ["| Method | Purpose |", "| --- | --- |"]
        output.append("### Methods")
        for key, value in methods:
            doc = inspect.getdoc(value) or ""
            table.append(
                f"| [{key}](#{name}.{key}) | {doc.splitlines()[0] if doc else key} |"
            )
        output.append("\n".join(table))
        for key, value in methods:
            output.extend(
                [
                    f'<span id="{name}.{key}"></span>',
                    f"### {key} {{.api-method-title}}",
                    '<div class="api-method-body" markdown="1">',
                    signature_block(value, f"{name}.{key}"),
                    render_doc(inspect.getdoc(value) or "", 4, owner=name),
                    "</div>",
                ]
            )
    output.append("</div>")
    return "\n\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    count = 0
    for path in sorted((ROOT / "docs").rglob("*.md")):
        if "_build" in path.parts:
            continue
        text = path.read_text()

        def replace(match):
            nonlocal count
            count += 1
            name = match[1]
            return f"<!-- BEGIN REFERENCE:{name} -->\n{reference(name)}\n<!-- END REFERENCE:{name} -->"

        updated = re.sub(
            r"<!-- BEGIN REFERENCE:(.*?) -->.*?<!-- END REFERENCE:\1 -->",
            replace,
            text,
            flags=re.S,
        )
        if args.check and updated != text:
            raise SystemExit(
                f"Stale reference: {path}; run scripts/build_api_reference.py"
            )
        if not args.check:
            path.write_text(updated)
    print(f"Verified {count} references against source signatures and docstrings.")


if __name__ == "__main__":
    main()
