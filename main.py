"""CLI chatbot with RAG support — entry point."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

HISTORIES_DIR = Path("histories")
EXPORTS_DIR = Path("exports")

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
  /model <name>      Switch to a different model
  /history save <name>  Save conversation to histories/<name>.json
  /history load <name>  Load conversation from histories/<name>.json
  /history list         List saved conversations
  /export [name]     Export conversation as Markdown to exports/
  /multiline         Toggle multiline input mode (submit with blank line)
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
    manager: ModelManager,
) -> tuple[bool, bool, str | None, str | None]:
    """
    Process a /command.
    Returns (should_continue, use_rag, new_system_prompt, new_model).
    None values mean unchanged.
    """
    parts = line.strip().split(None, 2)
    cmd = parts[0].lower()

    if cmd == "/quit":
        console.print("[dim]Goodbye.[/dim]")
        return False, use_rag, None, None

    if cmd == "/clear":
        history.clear()
        console.print("[dim]Conversation history cleared.[/dim]")
        return True, use_rag, None, None

    if cmd == "/model":
        if len(parts) < 2:
            # Show current model info
            console.print(
                f"[bold]Model:[/bold] {escape(model_name)}  "
                f"[bold]Backend:[/bold] {escape(backend_name)}"
            )
            return True, use_rag, None, None
        # Switch model
        new_model = parts[1].strip()
        try:
            available = manager.list_models()
        except RuntimeError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            return True, use_rag, None, None
        if backend_name == "ollama" and new_model not in available:
            console.print(
                f"[red]Model [bold]{escape(new_model)}[/bold] is not installed. "
                f"Run `python main.py models pull {escape(new_model)}` first.[/red]"
            )
            return True, use_rag, None, None
        console.print(f"[green]Switched to model [bold]{escape(new_model)}[/bold].[/green]")
        return True, use_rag, None, new_model

    if cmd == "/help":
        console.print(Panel(HELP_TEXT, border_style="dim"))
        return True, use_rag, None, None

    if cmd == "/system":
        if len(parts) < 2:
            # Show current system prompt
            sys_msg = next((m["content"] for m in history if m["role"] == "system"), None)
            if sys_msg:
                console.print(f"[bold]System prompt:[/bold] {escape(sys_msg)}")
            else:
                console.print("[dim]No system prompt set.[/dim]")
            return True, use_rag, None, None
        new_prompt = line.strip()[len("/system"):].strip()
        # Replace existing system message or prepend a new one
        for msg in history:
            if msg["role"] == "system":
                msg["content"] = new_prompt
                break
        else:
            history.insert(0, {"role": "system", "content": new_prompt})
        console.print(f"[green]System prompt updated.[/green]")
        return True, use_rag, new_prompt, None

    if cmd == "/rag":
        if rag is None:
            console.print("[red]RAG is not enabled. Start chat with --rag flag.[/red]")
            return True, use_rag, None, None

        sub = parts[1].lower() if len(parts) > 1 else ""

        if sub == "status":
            stats = rag.status()
            console.print(
                f"[bold]RAG store:[/bold] {stats['total_chunks']} chunks  "
                f"[bold]Sources:[/bold] {len(stats['sources'])}"
            )
            for src in stats["sources"]:
                console.print(f"  [dim]{escape(src)}[/dim]")
            return True, use_rag, None, None

        if sub == "add":
            if len(parts) < 3:
                console.print("[red]Usage: /rag add <path>[/red]")
                return True, use_rag, None, None
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
            return True, use_rag, None, None

        console.print("[red]Unknown /rag sub-command. Try: /rag add <path>, /rag status[/red]")
        return True, use_rag, None, None

    if cmd == "/history":
        sub = parts[1].lower() if len(parts) > 1 else ""

        if sub == "list":
            HISTORIES_DIR.mkdir(exist_ok=True)
            files = sorted(HISTORIES_DIR.glob("*.json"))
            if not files:
                console.print("[dim]No saved conversations.[/dim]")
            else:
                table = Table(title="Saved Conversations", header_style="bold magenta")
                table.add_column("Name", style="cyan")
                table.add_column("Messages", style="dim")
                for f in files:
                    try:
                        msgs = json.loads(f.read_text())
                        table.add_row(f.stem, str(len(msgs)))
                    except Exception:
                        table.add_row(f.stem, "[red]error[/red]")
                console.print(table)
            return True, use_rag, None, None

        if sub == "save":
            if len(parts) < 3:
                console.print("[red]Usage: /history save <name>[/red]")
                return True, use_rag, None, None
            name = parts[2].strip()
            HISTORIES_DIR.mkdir(exist_ok=True)
            dest = HISTORIES_DIR / f"{name}.json"
            dest.write_text(json.dumps(history, indent=2, ensure_ascii=False))
            console.print(f"[green]Saved {len(history)} messages to [bold]{escape(str(dest))}[/bold].[/green]")
            return True, use_rag, None, None

        if sub == "load":
            if len(parts) < 3:
                console.print("[red]Usage: /history load <name>[/red]")
                return True, use_rag, None, None
            name = parts[2].strip()
            src = HISTORIES_DIR / f"{name}.json"
            if not src.exists():
                console.print(f"[red]File not found: {escape(str(src))}[/red]")
                return True, use_rag, None, None
            try:
                loaded = json.loads(src.read_text())
                history.clear()
                history.extend(loaded)
                console.print(f"[green]Loaded {len(history)} messages from [bold]{escape(str(src))}[/bold].[/green]")
            except Exception as exc:
                console.print(f"[red]Failed to load: {escape(str(exc))}[/red]")
            return True, use_rag, None, None

        console.print("[red]Unknown /history sub-command. Try: save, load, list[/red]")
        return True, use_rag, None, None

    if cmd == "/multiline":
        # Toggled externally via the chat loop; signal via a special return value
        return True, use_rag, "__toggle_multiline__", None

    if cmd == "/export":
        if not history:
            console.print("[yellow]Nothing to export — conversation is empty.[/yellow]")
            return True, use_rag, None, None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = parts[1].strip() if len(parts) > 1 else timestamp
        EXPORTS_DIR.mkdir(exist_ok=True)
        dest = EXPORTS_DIR / f"{name}.md"
        lines = [
            f"# Chat Export — {model_name}",
            f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
        ]
        role_labels = {"system": "⚙️ System", "user": "**You**", "assistant": f"**{model_name}**"}
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            label = role_labels.get(role, role.capitalize())
            lines.append(f"### {label}")
            lines.append("")
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")
        dest.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]Exported {len(history)} messages to [bold]{escape(str(dest))}[/bold].[/green]")
        return True, use_rag, None, None

    console.print(f"[red]Unknown command: {escape(cmd)}. Type /help for help.[/red]")
    return True, use_rag, None, None


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
    multiline_mode = False

    if system:
        history.append({"role": "system", "content": system})

    # ---- conversation loop ------------------------------------------------
    while True:
        try:
            if multiline_mode:
                prompt = "[bold blue]You (multiline)[/bold blue] [dim]↵ blank line to send[/dim]\n"
                console.print(prompt, end="")
                lines = []
                while True:
                    line = console.input("[dim]...[/dim] ")
                    if line == "" and lines:
                        break
                    if line.startswith("/"):
                        # Allow slash commands even in multiline mode
                        user_input = line
                        lines = None
                        break
                    lines.append(line)
                if lines is None:
                    pass  # fall through to slash command handling below
                else:
                    user_input = "\n".join(lines)
            else:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Interrupted. Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            should_continue, use_rag, sys_or_signal, new_model = handle_slash_command(
                user_input, history, model, backend, rag, use_rag, manager
            )
            if not should_continue:
                break
            if sys_or_signal == "__toggle_multiline__":
                multiline_mode = not multiline_mode
                state = "[green]ON[/green]" if multiline_mode else "[yellow]OFF[/yellow]"
                console.print(f"[dim]Multiline mode: {state}. {'Submit with a blank line.' if multiline_mode else ''}[/dim]")
            if new_model:
                model = new_model
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
        token_stats: dict = {}
        try:
            for token in manager.generate(history, model=model, stream=True):
                if isinstance(token, dict) and token.get("__tokens__"):
                    token_stats = token
                else:
                    console.print(token, end="", markup=False)
                    full_response += token
            console.print()  # newline after stream
        except Exception as exc:
            console.print(f"\n[red]Error generating response: {escape(str(exc))}[/red]")
            history.pop()  # remove failed user message
            continue

        history.append({"role": "assistant", "content": full_response})

        dim_parts = []
        if num_chunks > 0:
            dim_parts.append(f"RAG: {num_chunks} chunk(s)")
        if token_stats:
            tps = token_stats.get("tokens_per_sec", 0.0)
            tps_str = f" @ {tps:.1f} tok/s" if tps > 0 else ""
            dim_parts.append(
                f"tokens: {token_stats['prompt']} in / {token_stats['completion']} out"
                f" / {token_stats['prompt'] + token_stats['completion']} total{tps_str}"
            )
        if dim_parts:
            console.print(f"[dim]  ({' | '.join(dim_parts)})[/dim]")


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
