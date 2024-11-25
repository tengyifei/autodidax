import os

import nox

nox.options.sessions = ["lint"]

# Define the minimal nox version required to run
nox.options.needs_version = ">= 2024.3.2"


@nox.session
def lint(session):
  session.install("ruff")
  session.run("ruff", "check")


@nox.session
def build_and_check_dists(session):
  session.install("build", "check-manifest >= 0.42", "twine")
  session.run("python", "-m", "build")
  session.run("python", "-m", "twine", "check", "dist/*")


@nox.session(python=["3.11", "3.12"])
def tests(session):
  session.install("pytest")
  build_and_check_dists(session)

  generated_files = os.listdir("dist/")
  generated_sdist = os.path.join("dist/", generated_files[1])

  session.install(generated_sdist)

  session.run("py.test", "tests/", *session.posargs)
