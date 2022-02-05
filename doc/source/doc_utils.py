# Adapted from the rtds_action sphinx extension.
# https://github.com/dfm/rtds-action/blob/main/src/rtds_action/rtds_action.py
# MIT License
# Copyright (c) 2021 Pysteps developers
# Copyright (c) 2020 Dan Foreman-Mackey

import os
import subprocess
import tarfile
import tempfile
from io import BytesIO
from zipfile import ZipFile

import requests


def current_release():
    """Returns current version as string from 'git describe' command."""
    from subprocess import check_output

    _version = check_output(["git", "describe", "--tags", "--always"])
    if _version:
        return _version.decode("utf-8")
    else:
        return "X.Y"


class ArtifactDownloader:
    """
    Class used to download Github artifacts.

    Notes
    -----
    Adapted from the rtds_action sphinx extension.
    https://github.com/dfm/rtds-action/blob/main/src/rtds_action/rtds_action.py
    """

    def __init__(self, repo, prefix="auto_examples-for-"):
        self.repo = repo
        self.prefix = prefix

    @staticmethod
    def current_commit_hash():
        try:
            cmd = "git rev-parse HEAD"
            git_hash = subprocess.check_output(cmd.split(" ")).strip().decode("ascii")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Error retrieving git hash. Return code {e.returncode}.\n" + e.stdout
            )
        return git_hash

    def available_artifacts(self):
        request_result = requests.get(
            f"https://api.github.com/repos/{self.repo}/actions/artifacts",
            params=dict(per_page=100),
        )

        if request_result.status_code != 200:
            raise RuntimeError(
                f"Error retrieving artifacts: ({request_result.status_code})"
            )
        request_result = request_result.json()
        artifacts = request_result.get("artifacts", [])
        return artifacts

    def current_hash_artifact(self):
        commit_hash = self.current_commit_hash()
        artifact_name = f"{self.prefix}{commit_hash}"

        artifacts = [
            artifact
            for artifact in self.available_artifacts()
            if artifact["name"] == artifact_name
        ]
        if len(artifacts) == 0:
            raise RuntimeError(f"Artifact `{artifact_name}` not found.")

        # Return the first artifact that matches the name. Ignore the rest.
        return artifacts[0]

    def download_artifact(
        self, token, dest_dir, retry=3, tar_filename="auto_examples.tar"
    ):
        # --- IMPORTANT ----------------------------------------------------------------
        #
        # This function uses a GitHub access token to download the artifact.
        #
        # To avoid leaking credentials:
        # - NEVER hardcode the token
        # - ALWAYS store the token value in environment variables as GitHub secrets
        #   or Read the docs environmental variables.
        # - NEVER print or log the token value for debugging
        # - NEVER print or log the request header since it contains the token.
        #
        # For testing:
        # - Generate a temporary token that expires the next day.
        # - Remove the temporary token from GitHub once the testing finished.
        # ------------------------------------------------------------------------------
        dest_dir = str(dest_dir)
        artifact = self.current_hash_artifact()
        for n in range(retry):
            # Access token needed to download artifacts
            # https://github.com/actions/upload-artifact/issues/51
            request_result = requests.get(
                artifact["archive_download_url"],
                headers={"Authorization": f"token {token}"},
            )

            if request_result.status_code != 200:
                print(
                    f"Can't download artifact ({request_result.status_code})."
                    f"Retry {n + 1}/{retry}"
                )
                continue

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_artifact_dir = os.path.join(tmp_dir, "artifact")
                with ZipFile(BytesIO(request_result.content)) as f:
                    f.extractall(path=local_artifact_dir)

                _tar_filename = os.path.join(local_artifact_dir, tar_filename)
                if not os.path.isfile(_tar_filename):
                    raise RuntimeError(f"{tar_filename} file not found in artifact.")

                my_tar = tarfile.open(_tar_filename)
                my_tar.extractall(dest_dir)
            return

        raise RuntimeError(f"Can't download artifact after {retry} retries.")
