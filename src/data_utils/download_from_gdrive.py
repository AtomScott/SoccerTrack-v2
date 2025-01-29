from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from rich.progress import Progress
from loguru import logger
import typer


def authenticate() -> GoogleDrive:
    """
    Authenticate with Google Drive using local credentials.
    
    Returns:
        GoogleDrive: Authenticated Google Drive instance.
    
    Raises:
        FileNotFoundError: If credentials.json is not found.
    """
    logger.info("Authenticating with Google Drive")
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile("credentials.json")
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def get_file_id_from_url(url: str) -> str:
    """
    Extract the file ID from a Google Drive URL.
    
    Args:
        url: Google Drive URL for a file or folder.
    
    Returns:
        str: The extracted file ID.
    
    Raises:
        ValueError: If the URL format is invalid or file ID cannot be extracted.
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "id" in query_params:
        return query_params["id"][0]
    else:
        # Extract the file ID from the URL path
        path_parts = parsed_url.path.split("/")
        if "folders" in path_parts:
            return path_parts[path_parts.index("folders") + 1]
        elif "file" in path_parts:
            return path_parts[path_parts.index("file") + 2]
        else:
            raise ValueError("Invalid Google Drive URL")


def download_file(drive: GoogleDrive, file_id: str, dest_path: Path) -> None:
    """
    Download a file or folder from Google Drive recursively.
    
    Args:
        drive: Authenticated Google Drive instance.
        file_id: ID of the file or folder to download.
        dest_path: Local path where the file or folder will be saved.
    
    Raises:
        Exception: If download fails or there are permission issues.
    """
    file_obj = drive.CreateFile({"id": file_id, "supportsAllDrives": True})
    file_obj.FetchMetadata(fields="title, mimeType")

    if file_obj["mimeType"] == "application/vnd.google-apps.folder":
        # It's a folder; download recursively
        folder_name = file_obj["title"]
        dest_folder = dest_path / folder_name
        dest_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading folder '{folder_name}' to '{dest_folder}'")

        # List all items inside the folder
        file_list = drive.ListFile(
            {
                "q": f"'{file_id}' in parents and trashed=false",
                "supportsAllDrives": True,
                "includeItemsFromAllDrives": True,
            }
        ).GetList()

        for item in file_list:
            download_file(drive, item["id"], dest_folder)
    else:
        # It's a file; download it
        file_name = file_obj["title"]
        dest_file = dest_path / file_name

        if dest_file.exists():
            logger.warning(f"Skipping existing file: '{dest_file}'")
            return

        logger.info(f"Downloading file '{file_name}' to '{dest_file}'")
        file_obj.GetContentFile(str(dest_file))


app = typer.Typer()


@app.command()
def download(
    url: str = typer.Argument(..., help="Google Drive file or folder URL"),
    destination: str = typer.Argument(..., help="Destination directory to save the files"),
) -> None:
    """
    Download files or folders from Google Drive.

    This command authenticates with Google Drive using local credentials and downloads
    the specified file or folder to the destination directory. For folders, it downloads
    all contents recursively.

    Args:
        url: Google Drive file or folder URL.
        destination: Destination directory to save the files.

    Raises:
        FileNotFoundError: If credentials.json is not found.
        ValueError: If the Google Drive URL is invalid.
        Exception: If download fails or there are permission issues.
    """
    drive = authenticate()
    file_id = get_file_id_from_url(url)
    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("[green]Downloading...", start=False)

        try:
            progress.start_task(task)
            download_file(drive, file_id, dest_path)
            progress.update(task, advance=1, completed=True)
            logger.info(f"Download complete. Files saved to '{dest_path}'")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise


if __name__ == "__main__":
    app()
