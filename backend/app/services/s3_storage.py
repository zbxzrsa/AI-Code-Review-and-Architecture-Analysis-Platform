"""S3/MinIO artifact storage service with presigned URL support."""
import os
import logging
from typing import Optional, Tuple
from datetime import datetime, timezone, timedelta
import json
import hashlib

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.client import Config

logger = logging.getLogger(__name__)


class S3ArtifactStorage:
    """S3/MinIO compatible artifact storage with presigned URL support."""

    def __init__(self):
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self.access_key_id = os.getenv("S3_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
        self.secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
        self.region_name = os.getenv("S3_REGION", "us-east-1")
        self.force_path_style = os.getenv("S3_FORCE_PATH_STYLE", "true").lower() == "true"
        self.secure = os.getenv("S3_SECURE", "false").lower() == "true"
        self.bucket_name = os.getenv("ARTIFACTS_BUCKET", "artifacts")

        # Presigned URL settings
        self.presign_download_enabled = os.getenv("ARTIFACTS_PRESIGN_DOWNLOAD_ENABLED", "true").lower() == "true"
        self.presign_download_ttl = int(os.getenv("ARTIFACTS_PRESIGN_DOWNLOAD_TTL", "86400"))  # 24h
        self.presign_upload_enabled = os.getenv("ARTIFACTS_PRESIGN_UPLOAD_ENABLED", "false").lower() == "true"
        self.presign_upload_ttl = int(os.getenv("ARTIFACTS_PRESIGN_UPLOAD_TTL", "900"))  # 15min

        # Initialize S3 client
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize boto3 S3 client."""
        try:
            session_kwargs = {}
            if self.access_key_id:
                session_kwargs["aws_access_key_id"] = self.access_key_id
            if self.secret_access_key:
                session_kwargs["aws_secret_access_key"] = self.secret_access_key

            session = boto3.Session(**session_kwargs)

            client_kwargs = {
                "region_name": self.region_name,
                "config": Config(s3={"addressing_style": "path" if self.force_path_style else "virtual"}),
            }

            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url

            self._client = session.client("s3", **client_kwargs)
            logger.info("S3 client initialized: endpoint=%s, bucket=%s", self.endpoint_url, self.bucket_name)
        except Exception as e:
            logger.error("Failed to initialize S3 client: %s", e)
            raise

    def upload_artifact(self, session_id: int, file_path: str, content: bytes, metadata: Optional[dict] = None) -> Tuple[str, str]:
        """Upload artifact to S3/MinIO.

        Args:
            session_id: Analysis session ID
            file_path: Relative file path (e.g., "src/main.py")
            content: File content as bytes
            metadata: Optional metadata dict

        Returns:
            Tuple of (object_url, etag)

        Raises:
            Exception if upload fails
        """
        if not self._client:
            raise RuntimeError("S3 client not initialized")

        # Generate S3 object key
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        file_hash = hashlib.md5(content).hexdigest()[:8]
        safe_file_path = file_path.replace("/", "_")
        s3_key = f"sessions/{session_id}/artifacts/{timestamp}/{safe_file_path}.{file_hash}.json"

        try:
            # Prepare metadata
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}

            # Upload
            response = self._client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content,
                ContentType="application/json",
                **extra_args
            )

            etag = response.get("ETag", "").strip('"')

            # Generate object URL
            if self.endpoint_url:
                # MinIO / custom endpoint
                protocol = "https" if self.secure else "http"
                object_url = f"{protocol}://{self.endpoint_url.split('://')[-1]}/{self.bucket_name}/{s3_key}"
            else:
                # AWS S3
                object_url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}"

            logger.info("Artifact uploaded: key=%s, etag=%s", s3_key, etag)
            return object_url, etag
        except (ClientError, BotoCoreError) as e:
            logger.error("S3 upload failed: %s", e)
            raise

    def generate_presigned_download_url(self, s3_key: str) -> Optional[str]:
        """Generate presigned download URL for artifact.

        Args:
            s3_key: S3 object key

        Returns:
            Presigned URL or None if presigned download disabled
        """
        if not self.presign_download_enabled:
            return None

        try:
            url = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=self.presign_download_ttl
            )
            logger.debug("Presigned download URL generated: key=%s", s3_key)
            return url
        except Exception as e:
            logger.error("Failed to generate presigned download URL: %s", e)
            return None

    def generate_presigned_upload_url(self, s3_key: str) -> Optional[str]:
        """Generate presigned upload URL for artifact.

        Args:
            s3_key: S3 object key

        Returns:
            Presigned URL or None if presigned upload disabled
        """
        if not self.presign_upload_enabled:
            return None

        try:
            url = self._client.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=self.presign_upload_ttl
            )
            logger.debug("Presigned upload URL generated: key=%s", s3_key)
            return url
        except Exception as e:
            logger.error("Failed to generate presigned upload URL: %s", e)
            return None


# Singleton instance
_storage_instance = None


def get_s3_storage() -> S3ArtifactStorage:
    """Get or create S3 storage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = S3ArtifactStorage()
    return _storage_instance
