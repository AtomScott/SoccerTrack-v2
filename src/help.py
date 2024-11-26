from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .main import load_commands


def print_help() -> None:
    """Print comprehensive help information about the application's usage and configuration."""
    console = Console()

    # Get available commands
    commands = load_commands()
    command_list = ", ".join(sorted(commands.keys()))

    # Main usage panel
    usage_text = (
        "This application uses [bold cyan]OmegaConf[/] for configuration management.\n\n"
        "Basic Usage:\n"
        "  [green]python -m src.main command=<command_name> (options)[/]\n\n"
        "Available Commands:\n"
        f"  [yellow]{command_list}[/]\n\n"
        "Example:\n"
        "  [green]python -m src.main command=log-string log_string.message='Hello World'[/]"
    )
    console.print(Panel(usage_text, title="📚 Usage", border_style="blue"))

    # Configuration table
    config_table = Table(title="⚙️ Configuration Management", show_header=True, header_style="bold magenta")
    config_table.add_column("Feature", style="cyan")
    config_table.add_column("Description", style="white")
    config_table.add_column("Example", style="green")

    config_table.add_row(
        "Base Config",
        "Default configuration in configs/default_config.yaml",
        "python -m src.main config_path=configs/default_config.yaml",
    )
    config_table.add_row(
        "CLI Overrides",
        "Override config values via command line",
        "command=log-string log_string.message='New message'",
    )

    # Commands table
    if commands:
        commands_table = Table(title="🛠️ Available Commands", show_header=True, header_style="bold magenta")
        commands_table.add_column("Command", style="cyan")
        commands_table.add_column("Description", style="white")

        for command_name, command_func in sorted(commands.items()):
            description = command_func.__doc__ or "No description available"
            # Clean up the docstring
            description = description.split("\n")[0].strip()
            commands_table.add_row(command_name, description)

        console.print(commands_table)
