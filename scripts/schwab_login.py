from settings import Settings, ensure_runtime_dirs
from schwab_client.auth import build_client


def main() -> None:
    settings = Settings()
    ensure_runtime_dirs(settings)
    build_client(settings)
    print(f"Token ready at: {settings.schwab_token_path}")


if __name__ == "__main__":
    main()
