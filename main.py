import click

from pong.service.app import create_app

app = create_app()


@click.command()
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default="8080")
def main(host: str, port: int):
    app.run(host=host, port=port)


if __name__ == '__main__':
    main()
