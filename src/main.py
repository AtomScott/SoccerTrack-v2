from importlib import import_module
from pathlib import Path
from types import ModuleType

from loguru import logger
from omegaconf import OmegaConf
from rich.traceback import install as install_rich_traceback

install_rich_traceback()

BASE_CONFIG_PATH = Path("configs/default_config.yaml")
COMMANDS_MODULE = "src"  # The module containing your commands


def load_commands() -> dict[str, callable]:
    """
    Dynamically load all commands from the commands module.

    Returns:
        dict[str, callable]: Dictionary mapping command names to their functions
    """
    try:
        commands_module: ModuleType = import_module(COMMANDS_MODULE)
        if not hasattr(commands_module, "__all__"):
            logger.warning(f"No __all__ defined in {COMMANDS_MODULE}")
            return {}

        commands: dict[str, callable] = {}
        for command_name in commands_module.__all__:
            command_func = getattr(commands_module, command_name, None)
            if callable(command_func):
                commands[command_name] = command_func
            else:
                logger.warning(f"Command '{command_name}' in __all__ is not callable")

        return commands

    except ImportError as e:
        logger.error(f"Failed to import commands module: {e}")
        return {}


def run_command(command: str, config: OmegaConf) -> None:
    """Run a command with the given configuration."""
    commands = load_commands()

    if command in commands:
        # Get the config section matching the command name (with underscores)
        config_section = config.get(command, {})
        commands[command](**config_section)
    else:
        available_commands = ", ".join(sorted(commands.keys()))
        logger.error(f"Unknown command: {command}")
        logger.info(f"Available commands: {available_commands}")


def main() -> None:
    """
    Main entry point for the application. Parses configuration and dispatches commands.
    """
    # Load and update the configuration
    cli_config = OmegaConf.from_cli()
    command = cli_config.get("command", "print_help")
    config_path = cli_config.get("config_path", BASE_CONFIG_PATH)

    base_config = OmegaConf.load(config_path)
    config = OmegaConf.merge(base_config, cli_config)

    run_command(command, config)


if __name__ == "__main__":
    main()
