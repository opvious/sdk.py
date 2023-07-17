import asyncio
import docopt
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
    {_COMMAND} register-notebook PATH [MODEL] [-n NAME] [-t TAGS]
    {_COMMAND} register-sources GLOB [-n NAME] [-t TAGS]
    {_COMMAND} (-h | --help)
    {_COMMAND} --version

Options:
    -n, --name NAME     Formulation name. By default this name is inferred
                        from the file's name, omitting the extension.
    -t, --tags TAGS     Comma-separated list of tags
    --version           Show SDK version
    -h, --help          Show this message
"""


async def _register_notebook(
    client: Client,
    path: str,
    model: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[str] = None,
) -> None:
    sn = load_notebook_models(path)
    if model is None:
        models = list(sn.__dict__.keys())
        if len(models) != 1:
            raise Exception(f"Notebook has 0 or 2+ models ({models})")
        model = models[0]
    obj = getattr(sn, model)
    if name is None:
        name = _default_name(path)
    await client.register_specification(
        obj.specification(),
        formulation_name=name,
        tag_names=tags.split() if tags else None,
    )
    print(f"Registered `{model}` as formulation `{name}`.")


async def _register_sources(
    client: Client,
    glob: str,
    name: Optional[str] = None,
    tags: Optional[str] = None,
) -> None:
    if name is None:
        name = _default_name(glob)
    await client.register_specification(
        LocalSpecification.globs(glob),
        formulation_name=name,
        tag_names=tags.split() if tags else None,
    )
    print(f"Registered sources as formulation `{name}`.")


def _default_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


async def _run(args: Mapping[str, Any]) -> None:
    client = Client.default()
    if args["register-notebook"]:
        await _register_notebook(
            client,
            args["PATH"],
            model=args["MODEL"],
            name=args["--name"],
            tags=args["--tags"],
        )
    elif args["register-sources"]:
        await _register_sources(
            client, args["GLOB"], name=args["--name"], tags=args["--tags"]
        )
    else:
        raise Exception("Unexpected command")


if __name__ == "__main__":
    argv = _COMMAND.split()[1:] + sys.argv[1:]
    args = docopt.docopt(_DOC, argv=argv, version=__version__)
    asyncio.run(_run(args))
