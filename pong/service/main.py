import click

from pong.service.app import create_app


@click.command()
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default="8080")
@click.option("--replay-buffer-path", default="replay-buffer", type=click.Path(dir_okay=True, file_okay=False))
def main(host: str, port: int, replay_buffer_path):
    app = create_app(replay_buffer_path)
    app.run(host=host, port=port)


if __name__ == '__main__':
    main()
