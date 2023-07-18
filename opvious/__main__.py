import asyncio
import dataclasses
import docopt
import json
import os.path
import sys
from typing import Any, Mapping, Optional

from .common import __version__
from .client import Client
from .specifications import load_notebook_models, LocalSpecification

_COMMAND = "python -m opvious"

_DOC = f"""Opvious Python CLI

Command line utilities for interacting with the Opvious API from Python. The
underlying client can be configured using the `OPVIOUS_ENDPOINT` and
`OPVIOUS_TOKEN` environment variables. Refer to the standard CLI
(https://www.npmjs.com/package/opvious-cli) for additional operations.

Usage:
    {_COMMAND} register-notebook PATH [MODEL] [-dn NAME] [-t TAGS]
    {_COMMAND} register-sources GLOB [-dn NAME] [-t TAGS]
    {_COMMAND} (-h | --help)
    {_COMMAND} --version

Options:
    -d, --dry-run       Validate the specification but do not store it on the
                        server.
    -n, --name NAME     Formulation name. By default this name is inferred
                        from the file's name, omitting the extension.
    -t, --tags TAGS     Comma-separated list of tags. By default only the
                        `latest` tag is added.
    --version           Show SDK version
    -h, --help          Show this message
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
        model_name: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        sn = load_notebook_models(path)
        if model_name is None:
            model_names = list(sn.__dict__.keys())
            if not self._dry_run and len(model_names) != 1:
                raise Exception(f"Notebook has 0 or 2+ models ({model_names})")
        else:
            model_names = [model_name]
        if name is None:
            name = _default_name(path)
        for model_name in model_names:
            model = getattr(sn, model_name)
            await self._handle(model.specification(), name)

    async def handle_sources(
        self, glob: str, name: Optional[str] = None
    ) -> None:
        if name is None:
            name = _default_name(glob)
        spec = LocalSpecification.globs(glob)
        await self._handle(spec, name)


def _default_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


async def _run(args: Mapping[str, Any]) -> None:
    handler = _SpecificationHandler(
        Client.default(), args["--tags"], args["--dry-run"]
    )
    if args["register-notebook"]:
        await handler.handle_notebook(
            args["PATH"],
            model_name=args["MODEL"],
            name=args["--name"],
        )
    elif args["register-sources"]:
        await handler.handle_sources(args["GLOB"], name=args["--name"])
    else:
        raise Exception("Unexpected command")


if __name__ == "__main__":
    argv = _COMMAND.split()[1:] + sys.argv[1:]
    args = docopt.docopt(_DOC, argv=argv, version=__version__)
    asyncio.run(_run(args))
