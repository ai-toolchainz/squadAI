import click

from .create_squad import create_squad


@click.group()
def squadai():
    """Top-level command group for squadai."""


@squadai.command()
@click.argument("project_name")
def create(project_name):
    """Create a new squad."""
    create_squad(project_name)


if __name__ == "__main__":
    squadai()
