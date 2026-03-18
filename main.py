"""CLI chatbot with RAG support — entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

from model_manager import MODEL_CATALOG, ModelManager
from rag_engine import RAGEngine

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BANNER = """[bold cyan]LLM CLI Chatbot[/bold cyan]  [dim]with RAG support[/dim]
Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit."""

HELP_TEXT = """\
[bold]Slash commands:[/bold]
  /quit              Exit the chatbot
  /clear             Clear conversation history
  /system <prompt>   Set a new system prompt
  /system            Show current system prompt
  /rag add <path>    Index a file or directory
  /rag status        Show RAG store statistics
  /model             Show current model info
  /help              Show this help message
"""


def print_banner() -> None:
    console.print(Panel(BANNER, border_style="cyan", padding=(0, 2)))


def choose_model_interactively(available: list[str]) -> str | None:
    """Show a table of available + catalog models and let user pick one."""
    table = Table(title="Available Models", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="cyan")
    table.add_column("Backend", style="green")
    table.add_column("Description")
    table.add_column("Status", style="bold")

    catalog_names = {m["name"] for m in MODEL_CATALOG}
    display: list[dict] = []

    # First: installed models present in catalog
    for m in MODEL_CATALOG:
        if m["name"] in available:
            display.append({**m, "installed": True})

    # Then: installed models NOT in catalog
    for name in available:
        if name not in catalog_names:
            display.append({"name": name, "backend": "ollama", "description": "-", "installed": True})

    # Finally: catalog models not installed
    for m in MODEL_CATALOG:
        if m["name"] not in available:
            display.append({**m, "installed": False})

    for i, m in enumerate(display, 1):
        status = "[green]installed[/green]" if m["installed"] else "[dim]not installed[/dim]"
        table.add_row(str(i), m["name"], m["backend"], m.get("description", "-"), status)

    console.print(table)

    if not display:
        console.print("[red]No models found. Run `python main.py models pull <name>` first.[/red]")
        return None

    installed = [m for m in display if m["installed"]]
    if not installed:
        console.print("[yellow]No models installed. Run `python main.py models pull <name>` first.[/yellow]")
        return None

    try:
        raw = console.input(
            f"[bold]Select a model[/bold] [dim](1-{len(display)}, or Enter for first installed)[/dim]: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if raw == "":
        return installed[0]["name"]

    try:
        idx = int(raw) - 1
        if 0 <= idx < len(display):
            chosen = display[idx]
            if not chosen["installed"] and chosen["backend"] == "ollama":
                console.print(f"[yellow]Model [bold]{chosen['name']}[/bold] is not installed. Pull it first.[/yellow]")
                return None
            return chosen["name"]
    except ValueError:
        pass

    console.print("[red]Invalid selection.[/red]")
    return None


def handle_slash_command(
    line: str,
    history: list[dict],
    model_name: str,
    backend_name: str,
    rag: RAGEngine | None,
    use_rag: bool,
) -> tuple[bool, bool, str | None]:
    """
    Process a /command.
    Returns (should_continue, use_rag, new_system_prompt).
    new_system_prompt is None if unchanged.
    """
    parts = line.strip().split(None, 2)
    cmd = parts[0].lower()

    if cmd == "/quit":
        console.print("[dim]Goodbye.[/dim]")
        return False, use_rag, None

    if cmd == "/clear":
        history.clear()
        console.print("[dim]Conversation history cleared.[/dim]")
        return True, use_rag, None

    if cmd == "/model":
        console.print(
            f"[bold]Model:[/bold] {escape(model_name)}  "
            f"[bold]Backend:[/bold] {escape(backend_name)}"
        )
        return True, use_rag, None

    if cmd == "/help":
        console.print(Panel(HELP_TEXT, border_style="dim"))
        return True, use_rag, None

    if cmd == "/system":
        if len(parts) < 2:
            # Show current system prompt
            sys_msg = next((m["content"] for m in history if m["role"] == "system"), None)
            if sys_msg:
                console.print(f"[bold]System prompt:[/bold] {escape(sys_msg)}")
            else:
                console.print("[dim]No system prompt set.[/dim]")
            return True, use_rag, None
        new_prompt = line.strip()[len("/system"):].strip()
        # Replace existing system message or prepend a new one
        for msg in history:
            if msg["role"] == "system":
                msg["content"] = new_prompt
                break
        else:
            history.insert(0, {"role": "system", "content": new_prompt})
        console.print(f"[green]System prompt updated.[/green]")
        return True, use_rag, new_prompt

    if cmd == "/rag":
        if rag is None:
            console.print("[red]RAG is not enabled. Start chat with --rag flag.[/red]")
            return True, use_rag, None

        sub = parts[1].lower() if len(parts) > 1 else ""

        if sub == "status":
            stats = rag.status()
            console.print(
                f"[bold]RAG store:[/bold] {stats['total_chunks']} chunks  "
                f"[bold]Sources:[/bold] {len(stats['sources'])}"
            )
            for src in stats["sources"]:
                console.print(f"  [dim]{escape(src)}[/dim]")
            return True, use_rag, None

        if sub == "add":
            if len(parts) < 3:
                console.print("[red]Usage: /rag add <path>[/red]")
                return True, use_rag, None
            target = Path(parts[2].strip()).expanduser()
            with console.status(f"[cyan]Indexing {escape(str(target))}…[/cyan]"):
                try:
                    if target.is_dir():
                        results = rag.add_directory(target)
                        total = sum(v for v in results.values() if v > 0)
                        console.print(f"[green]Indexed {len(results)} files, {total} chunks.[/green]")
                    else:
                        count = rag.add_document(target)
                        console.print(f"[green]Indexed {escape(target.name)}: {count} chunks.[/green]")
                    use_rag = True
                except Exception as exc:
                    console.print(f"[red]Error: {escape(str(exc))}[/red]")
            return True, use_rag, None

        console.print("[red]Unknown /rag sub-command. Try: /rag add <path>, /rag status[/red]")
        return True, use_rag, None

    console.print(f"[red]Unknown command: {escape(cmd)}. Type /help for help.[/red]")
    return True, use_rag, None


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """LLM CLI Chatbot with RAG support."""


# ---------- chat -----------------------------------------------------------

@cli.command()
@click.option("--model", "-m", default=None, help="Model name to use.")
@click.option(
    "--backend",
    default="ollama",
    show_default=True,
    type=click.Choice(["ollama", "api"]),
    help="Model backend.",
)
@click.option(
    "--api-provider",
    default="xai",
    show_default=True,
    type=click.Choice(["xai", "groq"]),
    help="API provider (only used when --backend=api).",
)
@click.option("--rag", "use_rag", is_flag=True, default=False, help="Enable RAG context injection.")
@click.option("--system", default=None, help="Optional system prompt.")
def chat(
    model: str | None,
    backend: str,
    api_provider: str,
    use_rag: bool,
    system: str | None,
) -> None:
    """Start an interactive chat session."""
    print_banner()

    manager = ModelManager(backend=backend, api_provider=api_provider)

    # Model selection
    if model is None:
        if backend == "ollama":
            try:
                available = manager.list_models()
            except RuntimeError as exc:
                console.print(f"[red]{escape(str(exc))}[/red]")
                sys.exit(1)
            model = choose_model_interactively(available)
            if model is None:
                sys.exit(1)
        else:
            # API: show catalog and ask
            available = manager.list_models()
            model = choose_model_interactively(available)
            if model is None:
                sys.exit(1)

    console.print(
        f"\n[bold green]Starting chat[/bold green] with [cyan]{escape(model)}[/cyan]"
        + (" [dim](RAG enabled)[/dim]" if use_rag else "")
        + "\n"
    )

    rag = RAGEngine() if use_rag else None
    history: list[dict] = []

    if system:
        history.append({"role": "system", "content": system})

    # ---- conversation loop ------------------------------------------------
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Interrupted. Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            should_continue, use_rag, _ = handle_slash_command(
                user_input, history, model, backend, rag, use_rag
            )
            if not should_continue:
                break
            # Re-create RAG engine if it was just enabled via /rag add
            if use_rag and rag is None:
                rag = RAGEngine()
            continue

        # Build message with optional RAG context
        user_content = user_input
        num_chunks = 0
        if use_rag and rag is not None:
            context, num_chunks = rag.build_context(user_input)
            if context:
                user_content = f"{context}\n\nUser question: {user_input}"

        history.append({"role": "user", "content": user_content})

        # Generate response
        console.print("[bold green]Assistant:[/bold green] ", end="")
        full_response = ""
        try:
            for token in manager.generate(history, model=model, stream=True):
                console.print(token, end="", markup=False)
                full_response += token
            console.print()  # newline after stream
        except Exception as exc:
            console.print(f"\n[red]Error generating response: {escape(str(exc))}[/red]")
            history.pop()  # remove failed user message
            continue

        history.append({"role": "assistant", "content": full_response})

        if num_chunks > 0:
            console.print(f"[dim]  (RAG: {num_chunks} relevant chunk(s) used)[/dim]")


# ---------- models ---------------------------------------------------------

@cli.group()
def models() -> None:
    """Manage Ollama models."""


@models.command("list")
@click.option("--backend", default="ollama", type=click.Choice(["ollama", "api"]))
@click.option("--api-provider", default="xai", type=click.Choice(["xai", "groq"]))
def models_list(backend: str, api_provider: str) -> None:
    """List installed/available models."""
    manager = ModelManager(backend=backend, api_provider=api_provider)
    try:
        model_names = manager.list_models()
    except RuntimeError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        sys.exit(1)

    if not model_names:
        console.print("[yellow]No models found.[/yellow]")
        return

    table = Table(title=f"Models ({backend})", header_style="bold magenta")
    table.add_column("Model", style="cyan")

    for name in model_names:
        table.add_row(name)

    console.print(table)


@models.command("pull")
@click.argument("model_name")
def models_pull(model_name: str) -> None:
    """Pull/download a model via Ollama."""
    manager = ModelManager(backend="ollama")

    console.print(f"[cyan]Pulling model:[/cyan] [bold]{escape(model_name)}[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Downloading…", total=None)
        try:
            for chunk in manager.pull_model(model_name):
                status = chunk.get("status", "")
                completed = chunk.get("completed", 0) or 0
                total = chunk.get("total", 0) or 0
                if total > 0:
                    progress.update(task_id, total=total, completed=completed, description=escape(status))
                else:
                    progress.update(task_id, description=escape(status))
        except RuntimeError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            sys.exit(1)

    console.print(f"[green]Done! Model [bold]{escape(model_name)}[/bold] is ready.[/green]")


# ---------- rag ------------------------------------------------------------

@cli.group()
def rag() -> None:
    """Manage the RAG document store."""


@rag.command("add")
@click.argument("path", type=click.Path(exists=True))
def rag_add(path: str) -> None:
    """Index a file or directory into the RAG store."""
    engine = RAGEngine()
    target = Path(path).expanduser().resolve()

    with console.status(f"[cyan]Indexing {escape(str(target))}…[/cyan]"):
        try:
            if target.is_dir():
                results = engine.add_directory(target)
                total = sum(v for v in results.values() if v > 0)
                console.print(f"[green]Indexed {len(results)} files, {total} chunks.[/green]")
                for src, cnt in results.items():
                    src_short = Path(src).name
                    if cnt >= 0:
                        console.print(f"  [dim]{escape(src_short)}: {cnt} chunks[/dim]")
                    else:
                        console.print(f"  [red]{escape(src_short)}: failed[/red]")
            else:
                count = engine.add_document(target)
                console.print(f"[green]Indexed [bold]{escape(target.name)}[/bold]: {count} chunks.[/green]")
        except Exception as exc:
            console.print(f"[red]Error: {escape(str(exc))}[/red]")
            sys.exit(1)


@rag.command("status")
def rag_status() -> None:
    """Show RAG store statistics."""
    engine = RAGEngine()
    stats = engine.status()
    console.print(f"[bold]Total chunks:[/bold] {stats['total_chunks']}")
    console.print(f"[bold]Sources:[/bold] {len(stats['sources'])}")
    for src in stats["sources"]:
        console.print(f"  [dim]{escape(src)}[/dim]")


@rag.command("clear")
def rag_clear() -> None:
    """Delete all documents from the RAG store."""
    if not click.confirm("This will delete all indexed documents. Continue?"):
        console.print("[dim]Aborted.[/dim]")
        return
    engine = RAGEngine()
    engine.clear()
    console.print("[green]RAG store cleared.[/green]")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
