from importlib import import_module
from pathlib import Path
from types import ModuleType

from loguru import logger
from omegaconf import OmegaConf
from rich.traceback import install as install_rich_traceback
import inspect
import re

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
    commands = load_commands()

    if command not in commands:
        available_commands = ", ".join(sorted(commands.keys()))
        logger.error(f"Unknown command: {command}")
        logger.info(f"Available commands: {available_commands}")
        return

    command_func = commands[command]
    config_section = config.get(command, {})

    try:
        command_func(**config_section)

    except TypeError as e:
        # Gather information about the function signature
        sig = inspect.signature(command_func)
        available_args = list(sig.parameters.keys())
        passed_args = list(config_section.keys())

        logger.error(f"Failed to run command '{command}': {e}")

        # 1) Show the available arguments
        logger.info(f"Available arguments for '{command}': {available_args}")

        # 2) Show the arguments that were actually passed
        logger.info(f"Provided arguments: {passed_args}")

        # Attempt to detect an unexpected argument by parsing the error message
        match = re.search(r"got an unexpected keyword argument '(.*)'", str(e))
        if match:
            unexpected_arg = match.group(1)
            logger.warning(f"Unexpected argument detected: '{unexpected_arg}'")

        # 3) Check for required (positional) arguments that were not passed
        required_args = [
            name
            for name, param in sig.parameters.items()
            if (param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY))
        ]
        missing_required_args = set(required_args) - set(passed_args)
        if missing_required_args:
            logger.warning(f"Missing required arguments for '{command}': {missing_required_args}")

    except Exception as e:
        # Catch any other errors your command might raise
        logger.exception(f"An error occurred while running '{command}': {e}")


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
