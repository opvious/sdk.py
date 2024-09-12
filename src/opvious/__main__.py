import asyncio
import dataclasses
import docopt
import json
import os.path
import sys
from typing import Any, Mapping, Optional

from . import __version__, Client, LocalSpecification, load_notebook_models
from .modeling import Model

_COMMAND = "python -m opvious"

_DOC = f"""Opvious Python CLI

Command line utilities for interacting with the Opvious API from Python. The
underlying client can be configured using the `OPVIOUS_ENDPOINT` and
`OPVIOUS_TOKEN` environment variables. Refer to the standard CLI
(https://www.npmjs.com/package/opvious-cli) for additional operations.

Usage:
    {_COMMAND} register-notebook PATH [MODEL]
        [-dn NAME] [-t TAGS] [--allow-empty]
    {_COMMAND} register-sources GLOB [-dn NAME] [-t TAGS]
    {_COMMAND} export-notebook-model PATH [MODEL] [-a PATH]
    {_COMMAND} (-h | --help)
    {_COMMAND} --version

Options:
    --allow-empty             Do not throw an error if no models were found in
                              a notebook. Requires `--dry-run` to be set.
    -a, --assembly-path PATH  Path where to store the exported model. Defaults
                              to the model's name with a `.proto` extension.
    -d, --dry-run             Validate the specification but do not store it on
                              the server. When this option is enabled,
                              notebooks can have more than one model.
    -h, --help                Show this message.
    -n, --name NAME           Formulation name. By default this name is
                              inferred from the file's name, omitting the
                              extension.
    -t, --tags TAGS           Comma-separated list of tags. By default only the
                              `latest` tag is added.
    --version                 Show SDK version.
"""


class _SpecificationHandler:
    def __init__(
        self, client: Client, tags: Optional[str] = None, dry_run=False
    ) -> None:
        self._client = client
        self._tags = tags.split() if tags else None
        self._dry_run = dry_run

    async def _handle(self, spec: LocalSpecification, name: str) -> None:
        if self._dry_run:
            annotated = await self._client.annotate_specification(spec)
            ann = annotated.annotation
            assert ann is not None
            if ann.issue_count > 0:
                raise Exception(
                    f"{ann.issue_count} issue(s) in specification: "
                    + json.dumps(
                        [
                            dataclasses.asdict(i)
                            for issues in ann.issues.values()
                            for i in issues
                        ]
                    )
                )
            print("Specification is valid")
        else:
            await self._client.register_specification(
                spec,
                formulation_name=name,
                tag_names=self._tags,
            )
            print(f"Registered specification in formulation {name}")

    async def handle_notebook(
        self,
        path: str,
        model_name: Optional[str],
        name: Optional[str],
        allow_empty: bool,
    ) -> None:
        models = _load_notebook_models(path, model_name)
        if self._dry_run:
            return
        _name, model = _singleton_model(models)
        await self._handle(model.specification(), name or _default_name(path))

    async def handle_sources(self, glob: str, name: Optional[str]) -> None:
        if name is None:
            name = _default_name(glob)
        spec = LocalSpecification.globs(glob)
        await self._handle(spec, name)


def _default_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load_notebook_models(
    path: str,
    model_name: Optional[str],
) -> dict[str, Model]:
    sn = load_notebook_models(path, allow_empty=True)
    if model_name is None:
        return {k: v for k, v in sn.__dict__.items() if isinstance(v, Model)}
    return {model_name: getattr(sn, model_name)}


def _singleton_model(models: dict[str, Model]) -> tuple[str, Model]:
    if len(models) != 1:
        raise Exception(
            "Notebook has 0 or 2+ models, please specify a model "
            "name to select one"
        )
    return next(iter(models.items()))


async def _export_notebook_model(
    client: Client,
    notebook_path: str,
    model_name: Optional[str] = None,
    export_path: Optional[str] = None,
) -> None:
    # TODO: Support transformations by accepting an additional variable name.
    models = _load_notebook_models(notebook_path, model_name)
    name, model = _singleton_model(models)
    if not export_path:
        export_path = f"{name}.proto"
    with open(export_path, "bw+") as writer:
        await client.export_specification(model.specification(), writer)


async def _run(args: Mapping[str, Any]) -> None:
    client = Client.from_environment()
    if not client:
        raise Exception("Missing OPVIOUS_ENDPOINT environment variable")

    if args["export-notebook-model"]:
        await _export_notebook_model(
            client,
            notebook_path=args["PATH"],
            model_name=args["MODEL"],
            export_path=args["--assembly-path"],
        )
        return

    handler = _SpecificationHandler(client, args["--tags"], args["--dry-run"])
    if args["register-notebook"]:
        await handler.handle_notebook(
            args["PATH"],
            model_name=args["MODEL"],
            name=args["--name"],
            allow_empty=args["--allow-empty"],
        )
    elif args["register-sources"]:
        await handler.handle_sources(args["GLOB"], name=args["--name"])
    else:
        raise Exception("Unexpected command")


if __name__ == "__main__":
    argv = _COMMAND.split()[1:] + sys.argv[1:]
    args = docopt.docopt(_DOC, argv=argv, version=__version__)
    asyncio.run(_run(args))
