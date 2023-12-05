# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a physical layer for remote files, accessed via S3.
"""

from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlparse

import uproot.extras
import uproot.source.http


class S3Source(uproot.source.http.HTTPSource):
    """
    Args:
        file_path (str): A URL of the file to open.
        endpoint: S3 endpoint (defaults to AWS)
        access_key: Access key of your S3 account
        secret_key: Secret key of your S3 account
        session_token: Session token of your S3 account
        secure: Flag to enable use of TLS
        http_client (urllib3.poolmanager.PoolManager): Instance of :doc:`urllib3.poolmanager.PoolManager`
        credentials (minio.credentials.Provider): Instance of :doc:`minio.credentials.Provider`
        options: See :doc:`uproot.source.http.HTTPSource.__init__`
    """

    def __init__(
        self,
        file_path: str,
        endpoint="s3.amazonaws.com",
        access_key=None,
        secret_key=None,
        session_token=None,
        secure=True,
        region=None,
        http_client=None,
        credentials=None,
        **options,
    ):
        Minio = uproot.extras.Minio_client()

        self._file_path = file_path
        if access_key is None:
            access_key = os.environ.get(
                "S3_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID", None)
            )
        if secret_key is None:
            secret_key = os.environ.get(
                "S3_SECRET_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", None)
            )
        if session_token is None:
            session_token = os.environ.get(
                "S3_SESSION_TOKEN", os.environ.get("AWS_SESSION_TOKEN", None)
            )
        if region is None:
            region = os.environ.get("AWS_DEFAULT_REGION", None)

        parsed_url = urlparse(file_path)

        bucket_name = parsed_url.netloc
        assert parsed_url.path[0] == "/"
        object_name = parsed_url.path[1:]

        parsed_query = dict(parse_qsl(parsed_url.query))
        # There is no standard scheme for s3:// URI query parameters,
        # but some are often introduced to support extra flexibility:
        if "endpoint" in parsed_query:
            endpoint = parsed_query["endpoint"]
        if "region" in parsed_query:
            region = parsed_query["region"]

        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
            secure=secure,
            region=region,
            http_client=http_client,
            credentials=credentials,
        )

        url = client.get_presigned_url("GET", bucket_name, object_name)

        super().__init__(url, **options)
